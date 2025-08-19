import os

import numpy as np

import pybamm
from pybamm.util import import_optional_dependency


class ScikitFemGenerator3D(pybamm.MeshGenerator):
    """
    A mesh generator that creates 3D tetrahedral meshes using scikit-fem.

    Parameters
    ----------
    geom_type : str
        The type of geometry to generate. Must be one of "pouch" for a rectangular
        prism, or "cylinder" for a cylindrical annulus.
    gen_params : dict
        A dictionary of geometry-specific parameters. for eg:

        - "h" : float, optional
            The target characteristic length of the mesh elements. A smaller 'h'
            results in a finer mesh and a more accurate solution, but
            increases computation time. Default is 0.3.
    """

    def __init__(self, geom_type, **gen_params):
        supported_geometries = ["pouch", "cylinder"]
        if geom_type not in supported_geometries:
            raise pybamm.GeometryError(
                f"geom_type must be one of {supported_geometries}, not '{geom_type}'"
            )
        super().__init__(ScikitFemSubMesh3D)
        self.geom_type = geom_type
        self.gen_params = gen_params

    def _make_pouch_mesh(self, x_lim, y_lim, z_lim, h):
        """
        Create a structured pouch mesh.

        Parameters
        ----------
        x_lim, y_lim, z_lim : tuple
            Domain limits for each dimension
        h : float
            Target mesh size

        Returns
        -------
        skfem.MeshTet
            Pouch mesh with proper boundary tags
        """
        skfem = import_optional_dependency("skfem")
        x_min, x_max = x_lim
        y_min, y_max = y_lim
        z_min, z_max = z_lim

        nx = max(2, int(np.ceil((x_max - x_min) / h)) + 1)
        ny = max(2, int(np.ceil((y_max - y_min) / h)) + 1)
        nz = max(2, int(np.ceil((z_max - z_min) / h)) + 1)

        mesh = skfem.MeshTet.init_tensor(
            np.linspace(x_min, x_max, nx),
            np.linspace(y_min, y_max, ny),
            np.linspace(z_min, z_max, nz),
        )

        bnd_facets = mesh.boundary_facets()
        midpoints = mesh.p[:, mesh.facets[:, bnd_facets]].mean(axis=1)

        boundaries = {
            "x_min": bnd_facets[np.isclose(midpoints[0], x_min)],
            "x_max": bnd_facets[np.isclose(midpoints[0], x_max)],
            "y_min": bnd_facets[np.isclose(midpoints[1], y_min)],
            "y_max": bnd_facets[np.isclose(midpoints[1], y_max)],
            "z_min": bnd_facets[np.isclose(midpoints[2], z_min)],
            "z_max": bnd_facets[np.isclose(midpoints[2], z_max)],
        }

        subdomains = {"default": np.arange(mesh.nelements)}
        return mesh.with_boundaries(boundaries).with_subdomains(subdomains)

    def _make_cylindrical_mesh(self, r_lim, z_lim, h):
        """
        Create a cylindrical annulus mesh.

        Parameters
        ----------
        r_lim : tuple
            Radial limits (inner radius, outer radius)
        z_lim : tuple
            Vertical limits (z_min, z_max)
        h : float
            Target mesh size

        Returns
        -------
        skfem.MeshTet
            Cylindrical mesh with proper boundary tags
        """
        skfem = import_optional_dependency("skfem")
        from scipy.spatial import Delaunay

        r_inner, r_outer = r_lim
        z_min, z_max = z_lim
        height = z_max - z_min

        n_radial = max(4, int((r_outer - r_inner) / h) + 1)
        n_theta_base = max(16, int(2 * np.pi * r_outer / h))
        r_coords = np.linspace(r_inner, r_outer, n_radial)
        points_list = []
        for r in r_coords:
            if r > 1e-9:
                n_theta = max(8, int(n_theta_base * (r / r_outer)))
                thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
                for theta in thetas:
                    points_list.append([r * np.cos(theta), r * np.sin(theta)])
        points_2d = np.array(points_list)
        tri_2d = Delaunay(points_2d)
        triangles_base = tri_2d.simplices

        n_z = max(5, int(height / h))
        z_coords = np.linspace(z_min, z_max, n_z)
        n_nodes_per_layer = points_2d.shape[0]
        nodes_3d = np.zeros((n_nodes_per_layer * n_z, 3))
        for i, z in enumerate(z_coords):
            start, end = i * n_nodes_per_layer, (i + 1) * n_nodes_per_layer
            nodes_3d[start:end, :2] = points_2d
            nodes_3d[start:end, 2] = z
        tetrahedra = []
        for i in range(n_z - 1):
            for tri in triangles_base:
                v = [node_idx + (i * n_nodes_per_layer) for node_idx in tri]
                v_top = [node_idx + ((i + 1) * n_nodes_per_layer) for node_idx in tri]
                tetrahedra.extend(
                    [
                        [v[0], v[1], v[2], v_top[2]],
                        [v[0], v[1], v_top[1], v_top[2]],
                        [v[0], v_top[0], v_top[1], v_top[2]],
                    ]
                )
        mesh = skfem.MeshTet(nodes_3d.T, np.array(tetrahedra).T)

        bottom_nodes = set(range(n_nodes_per_layer))
        top_nodes = set(range(n_nodes_per_layer * (n_z - 1), n_nodes_per_layer * n_z))

        points_2d_r = np.linalg.norm(points_2d, axis=1)
        tol = h * 0.1
        inner_nodes_2d = np.where(np.isclose(points_2d_r, r_inner, atol=tol))[0]
        outer_nodes_2d = np.where(np.isclose(points_2d_r, r_outer, atol=tol))[0]

        inner_wall_nodes = {
            n + (i * n_nodes_per_layer) for i in range(n_z) for n in inner_nodes_2d
        }
        outer_wall_nodes = {
            n + (i * n_nodes_per_layer) for i in range(n_z) for n in outer_nodes_2d
        }

        bottom_facets, top_facets, inner_facets, outer_facets = [], [], [], []

        for facet_idx in mesh.boundary_facets():
            facet_nodes = set(mesh.facets[:, facet_idx])
            if facet_nodes.issubset(bottom_nodes):
                bottom_facets.append(facet_idx)
            elif facet_nodes.issubset(top_nodes):
                top_facets.append(facet_idx)
            elif facet_nodes.issubset(inner_wall_nodes):
                inner_facets.append(facet_idx)
            elif facet_nodes.issubset(outer_wall_nodes):
                outer_facets.append(facet_idx)

        boundaries = {}
        if len(bottom_facets) > 0:
            boundaries["z_min"] = np.array(bottom_facets)
        if len(top_facets) > 0:
            boundaries["z_max"] = np.array(top_facets)
        if len(inner_facets) > 0:
            boundaries["r_min"] = np.array(inner_facets)
        if len(outer_facets) > 0:
            boundaries["r_max"] = np.array(outer_facets)

        return mesh.with_boundaries(boundaries)

    def __call__(self, lims, npts):
        """
        Main entry point called by PyBaMM's Discretisation.

        Parameters
        ----------
        lims : dict
            Dictionary of domain limits
        npts : dict
            Dictionary of target number of points

        Returns
        -------
        ScikitFemSubMesh3D
            Generated 3D submesh
        """
        h = float(self.gen_params.get("h", 0.3))

        lims_dict = {}
        for var, lim in lims.items():
            if isinstance(var, pybamm.SpatialVariable):
                lims_dict[var.name] = (lim["min"], lim["max"])
            else:
                lims_dict[var] = (float(lim["min"]), float(lim["max"]))

        if self.geom_type == "pouch":
            x_lim = lims_dict["x"]
            y_lim = lims_dict["y"]
            z_lim = lims_dict["z"]
            coord_sys = "cartesian"
            mesh = self._make_pouch_mesh(x_lim, y_lim, z_lim, h)

        elif self.geom_type == "cylinder":
            if "r" in lims_dict:
                r_lim = lims_dict["r"]
            else:
                r_lim = lims_dict["r_macro"]

            z_lim = lims_dict["z"]
            coord_sys = "cylindrical polar"
            mesh = self._make_cylindrical_mesh(r_lim, z_lim, h)
        else:
            raise ValueError(f"Unknown geom_type: {self.geom_type}")  # pragma: no cover

        if mesh is None:
            raise pybamm.DiscretisationError(
                f"Mesh generation failed for {self.geom_type}"
            )  # pragma: no cover

        nodes = mesh.p.T
        elements = mesh.t.T
        submesh = self.submesh_type(mesh, nodes, elements, coord_sys)
        return submesh


class UserSuppliedSubmesh3D(pybamm.MeshGenerator):
    """
    A mesh generator that loads a 3D mesh from an external file (e.g., VTK).

    This allows using high-quality meshes from any external tool. The mesh
    file must contain integer tags identifying the different physical domains
    and boundaries. This generator works directly with your existing
    'ScikitFemSubMesh3D' and 'ScikitFiniteElement3D' classes.

    Parameters
    ----------
    file_path : str
        Path to the mesh file.
    boundary_mapping : dict
        Maps PyBaMM boundary names to the integer tags in the mesh file.
    domain_mapping : dict
        Maps PyBaMM domain names to the integer tags in the mesh file.
    domain_tag_name : str, optional
        The name of the data array in the mesh file for domain tags.
    boundary_tag_name : str, optional
        The name of the data array in the mesh file for boundary tags.
    coord_sys : str, optional
        The coordinate system ("cartesian" or "cylindrical polar").

    Notes
    -----
    The external mesh file must meet the following criteria to be compatible:

        1.  **Element Types**: The 3D volume must be meshed with 4-node
            **tetrahedral** elements. The 2D boundaries must be meshed with
            3-node **triangular** elements.

        2.  **Physical Tags**: The file must contain integer tags to identify
            physical groups.
            - Each tetrahedron in the volume must be tagged with an integer
            that corresponds to a PyBaMM domain.
            - Each triangle on a boundary must be tagged with an integer that
            corresponds to a PyBaMM boundary.

        3.  **File Format**: Any format supported by the `meshio` library is
            acceptable. For maximum reliability in preserving physical tags,
            the **Gmsh `.msh` Version 2.2 (ASCII)** format is recommended.
    """

    def __init__(
        self,
        file_path,
        boundary_mapping,
        domain_mapping,
        domain_tag_name=None,
        boundary_tag_name=None,
        coord_sys=None,
    ):
        super().__init__(ScikitFemSubMesh3D)
        self.file_path = file_path
        self.boundary_mapping = boundary_mapping
        self.domain_mapping = domain_mapping
        self.domain_tag_name = domain_tag_name
        self.boundary_tag_name = boundary_tag_name

        if coord_sys:
            self.coord_sys = coord_sys
        else:
            if any("r_min" in k or "r_max" in k for k in boundary_mapping.keys()):
                self.coord_sys = "cylindrical polar"
            else:
                self.coord_sys = "cartesian"

    def __call__(self, lims, npts):
        skfem_mesh = ScikitFemSubMesh3D.load_mesh_from_file(
            self.file_path,
            self.boundary_mapping,
            self.domain_mapping,
            self.domain_tag_name,
            self.boundary_tag_name,
        )
        nodes = skfem_mesh.p.T
        elements = skfem_mesh.t.T
        return self.submesh_type(skfem_mesh, nodes, elements, self.coord_sys)


class ScikitFemSubMesh3D(pybamm.SubMesh):
    """
    A 3D submesh class for unstructured tetrahedral meshes generated by scikit-fem.

    Parameters
    ----------
    skfem_mesh : skfem.MeshTet
        The scikit-fem mesh object
    nodes : array_like
        Array of node coordinates (npts, 3)
    elements : array_like
        Array of element connectivity (nelements, 4)
    coord_sys : str
        The coordinate system of the submesh
    tabs : dict, optional
        Information about tabs (unused in 3D)
    """

    def __init__(self, skfem_mesh, nodes, elements, coord_sys, tabs=None):
        super().__init__()
        skfem = import_optional_dependency("skfem")

        self._skfem_mesh = skfem_mesh
        self.nodes = nodes
        self.elements = elements
        self.coord_sys = coord_sys
        self.tabs = tabs
        self.dimension = 3
        self.npts = nodes.shape[0]
        self.nelements = elements.shape[0]
        self.edges = self._skfem_mesh.edges.T

        self.basis = skfem.InteriorBasis(self._skfem_mesh, skfem.ElementTetP1())
        self.facet_basis = skfem.FacetBasis(self._skfem_mesh, self.basis.elem)
        self.element = skfem.ElementTetP1()

        if (
            hasattr(self._skfem_mesh, "boundaries")
            and self._skfem_mesh.boundaries is not None
        ):
            for name in self._skfem_mesh.boundaries:
                facet_basis = skfem.FacetBasis(
                    self._skfem_mesh,
                    self.basis.elem,
                    facets=self._skfem_mesh.boundaries[name],
                )
                setattr(self, f"{name}_basis", facet_basis)

                dofs = self.basis.get_dofs(name).all()
                setattr(self, f"{name}_dofs", dofs)

                normals = facet_basis.normals
                setattr(self, f"{name}_normals", normals)

        self._compute_element_volumes()
        self._compute_element_centroids()
        self.internal_boundaries = []

    def _compute_element_volumes(self):
        """Compute volumes of all tetrahedral elements."""
        vertices = self.nodes[self.elements]
        v1 = vertices[:, 1] - vertices[:, 0]
        v2 = vertices[:, 2] - vertices[:, 0]
        v3 = vertices[:, 3] - vertices[:, 0]
        self.element_volumes = np.abs(np.einsum("ij,ij->i", np.cross(v1, v2), v3)) / 6

    def _compute_element_centroids(self):
        """Compute centroids of all tetrahedral elements."""
        self.element_centroids = np.mean(self.nodes[self.elements], axis=1)

    def to_json(self):
        """Convert mesh to JSON format."""
        return {
            "mesh_type": self.__class__.__name__,
            "nodes": self.nodes.tolist(),
            "elements": self.elements.tolist(),
            "coord_sys": self.coord_sys,
            "element_volumes": self.element_volumes.tolist(),
            "element_centroids": self.element_centroids.tolist(),
        }

    @classmethod
    def _from_json(cls, json_dict):
        """Create mesh from JSON format."""
        skfem = import_optional_dependency("skfem")
        nodes = np.array(json_dict["nodes"])
        elements = np.array(json_dict["elements"])
        mesh = skfem.MeshTet(nodes.T, elements.T)
        return cls(mesh, nodes, elements, json_dict["coord_sys"])

    @staticmethod
    def load_mesh_from_file(
        file_path,
        boundary_mapping,
        domain_mapping,
        domain_tag_name=None,
        boundary_tag_name=None,
    ):
        """
        Loads a generic mesh file and converts it to a scikit-fem Mesh object.

        This function is designed to work with meshes from most software packages
        including GMSH, Ansys, Abaqus, FEniCS, etc. It automatically detects
        common tag naming conventions or allows manual specification.

        Parameters
        ----------
        file_path : str
            The path to the mesh file (It supports .msh, .xdmf, may also work with others but not tested).
        boundary_mapping : dict
            Maps PyBaMM boundary names to integer tags (e.g., {"r_inner": 1}).
        domain_mapping : dict
            Maps PyBaMM domain names to integer tags (e.g., {"current collector": 5}).
        domain_tag_name : str, optional
            The name of the cell data array that contains domain tags for elements.
        boundary_tag_name : str, optional
            The name of the cell data array that contains boundary tags for facets.
        """
        meshio = pybamm.util.import_optional_dependency("meshio")
        skfem = pybamm.util.import_optional_dependency("skfem")

        tested_formats = [".msh", ".xdmf"]
        _, file_ext = os.path.splitext(file_path)
        if file_ext not in tested_formats:  # pragma: no cover
            pybamm.logger.warning(
                f"File format '{file_ext}' has not been explicitly tested and may not "
                "work correctly. Recommended formats are .msh and .xdmf."
            )

        try:
            m = meshio.read(file_path)
        except Exception as e:
            raise pybamm.GeometryError(
                f"Could not read mesh file '{file_path}': {e}"
            ) from e

        nodes = m.points
        if nodes.shape[1] != 3:
            raise pybamm.GeometryError(
                f"Mesh must be 3D, got points with shape {nodes.shape}"
            )

        def find_tag_name(
            user_provided_name, default_candidates, available_data, context=""
        ):
            """
            Enhanced tag name finder that works with most mesh formats.
            """
            # If user explicitly provided a name, use it
            if user_provided_name:
                if user_provided_name in available_data:
                    return user_provided_name
                else:
                    available_keys = list(available_data.keys())
                    raise pybamm.GeometryError(
                        f"User-specified {context} tag name '{user_provided_name}' not found. "
                        f"Available arrays: {available_keys}"
                    )

            for candidate in default_candidates:
                if candidate in available_data:
                    return candidate

            integer_arrays = [
                name
                for name, data in available_data.items()
                if np.issubdtype(data.dtype, np.integer)
            ]

            if len(integer_arrays) == 1:
                return integer_arrays[0]

            priority_patterns = ["physical", "tag", "id", "group", "material", "region"]
            for pattern in priority_patterns:
                for name in integer_arrays:
                    if pattern.lower() in name.lower():
                        return name

            raise pybamm.GeometryError(
                f"Could not automatically detect {context} tag array in '{file_path}'. "
                f"Available arrays: {list(available_data.keys())}. "
                f"Integer arrays found: {integer_arrays}. "
                f"Please specify manually using '{context.replace(' ', '_')}_tag_name'."
            )

        try:
            tet_cells = m.get_cells_type("tetra")
            if len(tet_cells) == 0:
                raise pybamm.GeometryError(
                    f"No tetrahedral elements found in '{file_path}'"
                )

            tet_cell_index = None
            for i, cell_block in enumerate(m.cells):
                if cell_block.type == "tetra":
                    tet_cell_index = i
                    break

            if tet_cell_index is None:
                raise pybamm.GeometryError("Could not find tetrahedra in cell list")

            tet_cell_data_dict = {}
            for name, data_list in m.cell_data.items():
                if len(data_list) > tet_cell_index:
                    data = data_list[tet_cell_index]
                    if len(data) == len(tet_cells):
                        tet_cell_data_dict[name] = data

            domain_tag_name = find_tag_name(
                domain_tag_name,
                [
                    "gmsh:physical",
                    "physical",
                    "PhysicalGroup",
                    "MaterialId",
                    "CellEntityIds",
                    "gmsh:geometrical",
                ],
                tet_cell_data_dict,
                "domain",
            )

            elements = tet_cells
            tet_cell_data = tet_cell_data_dict[domain_tag_name]

            subdomains = {}
            for name, tag in domain_mapping.items():
                matching_elements = np.where(tet_cell_data == tag)[0]
                if len(matching_elements) == 0:
                    pybamm.logger.warning(
                        f"No elements found for domain '{name}' with tag {tag}"
                    )
                else:
                    subdomains[name] = matching_elements
                    pybamm.logger.info(
                        f"Found {len(matching_elements)} elements for domain '{name}' (tag {tag})"
                    )

        except Exception as e:
            raise pybamm.GeometryError(
                f"Failed to extract tetrahedral elements from '{file_path}': {e}"
            ) from e

        skfem_mesh = skfem.MeshTet(nodes.T, elements.T)

        boundaries = {}
        try:
            tri_cells = m.get_cells_type("triangle")

            if len(tri_cells) > 0:
                tri_cell_indices = []
                for i, cell_block in enumerate(m.cells):
                    if cell_block.type == "triangle":
                        tri_cell_indices.append(i)

                # Combine all triangle cell data
                all_tri_cells = []
                all_tri_data = []

                for idx in tri_cell_indices:
                    block_cells = m.cells[idx].data
                    all_tri_cells.extend(block_cells)

                    for name, data_list in m.cell_data.items():
                        if len(data_list) > idx:
                            if name not in [arr_name for arr_name, _ in all_tri_data]:
                                all_tri_data.append((name, []))
                            for arr_name, arr_data in all_tri_data:
                                if arr_name == name:
                                    arr_data.extend(data_list[idx])
                                    break

                all_tri_cells = np.array(all_tri_cells)

                tri_cell_data_dict = {}
                for name, data in all_tri_data:
                    tri_cell_data_dict[name] = np.array(data)

                if tri_cell_data_dict:
                    # Find boundary tag name
                    boundary_tag_name = find_tag_name(
                        boundary_tag_name,
                        [
                            "gmsh:physical",
                            "physical",
                            "PhysicalGroup",
                            "FaceEntityIds",
                            "gmsh:geometrical",
                        ],
                        tri_cell_data_dict,
                        "boundary",
                    )

                    tri_cell_data = tri_cell_data_dict[boundary_tag_name]

                    # Map triangle facets to mesh boundary facets
                    bnd_facets_indices = skfem_mesh.boundary_facets()
                    bnd_facets_nodes = skfem_mesh.facets[:, bnd_facets_indices]

                    # Create mapping from node sets to facet indices
                    facet_map = {}
                    for i in range(len(bnd_facets_indices)):
                        node_set = frozenset(bnd_facets_nodes[:, i])
                        facet_map[node_set] = bnd_facets_indices[i]

                    # Find boundary facets for each tag
                    for name, tag in boundary_mapping.items():
                        tagged_triangles = all_tri_cells[tri_cell_data == tag]
                        facet_indices = []

                        for tri in tagged_triangles:
                            tri_set = frozenset(tri)
                            if tri_set in facet_map:
                                facet_indices.append(facet_map[tri_set])

                        if facet_indices:
                            boundaries[name] = np.array(facet_indices)
                            pybamm.logger.info(
                                f"Found {len(facet_indices)} boundary facets for '{name}' (tag {tag})"
                            )
                        else:
                            pybamm.logger.warning(
                                f"No boundary facets found for '{name}' with tag {tag}"
                            )

        except Exception as e:
            pybamm.logger.warning(
                f"Could not extract boundary information from '{file_path}': {e}"
            )

        if boundaries:
            skfem_mesh = skfem_mesh.with_boundaries(boundaries)
        if subdomains:
            skfem_mesh = skfem_mesh.with_subdomains(subdomains)

        pybamm.logger.info(
            f"Successfully loaded mesh: {len(nodes)} nodes, {len(elements)} elements, "
            f"{len(boundaries)} boundary groups, {len(subdomains)} subdomains"
        )

        return skfem_mesh
