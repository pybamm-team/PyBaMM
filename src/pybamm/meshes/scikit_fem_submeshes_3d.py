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
