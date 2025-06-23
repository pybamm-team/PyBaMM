import numpy as np

import pybamm
from pybamm.util import import_optional_dependency


def laplacian_smooth(mesh, boundary_dofs, iterations=5):
    """
    Improves mesh quality using Laplacian smoothing while keeping boundary nodes fixed.

    Parameters
    ----------
    mesh : skfem.MeshTet
        The tetrahedral mesh to smooth
    boundary_dofs : array_like
        Indices of boundary nodes to keep fixed
    iterations : int
        Number of smoothing iterations

    Returns
    -------
    skfem.MeshTet
        Smoothed mesh
    """
    skfem = import_optional_dependency("skfem")
    p = mesh.p.copy()
    edges = mesh.edges

    adjacency = [[] for _ in range(p.shape[1])]
    for i in range(edges.shape[1]):
        u, v = edges[:, i]
        adjacency[u].append(v)
        adjacency[v].append(u)

    interior_nodes = np.ones(p.shape[1], dtype=bool)
    interior_nodes[np.unique(boundary_dofs)] = False
    interior_indices = np.where(interior_nodes)[0]

    for _ in range(iterations):
        p_new = p.copy()
        for i in interior_indices:
            neighbors = adjacency[i]
            if neighbors:
                p_new[:, i] = np.mean(p[:, neighbors], axis=1)
        p = p_new

    smoothed_mesh = skfem.MeshTet(p, mesh.t)

    return smoothed_mesh


class ScikitFemGenerator3D(pybamm.MeshGenerator):
    """
    A mesh generator that creates 3D tetrahedral meshes using scikit-fem.

    Parameters
    ----------
    geom_type : str
        Type of geometry to generate ('box', 'cylinder', 'spiral')
    **gen_params : dict
        Geometry-specific parameters
    """

    def __init__(self, geom_type, **gen_params):
        supported_geometries = ["box", "cylinder"]
        if geom_type not in supported_geometries:
            raise pybamm.GeometryError(
                f"geom_type must be one of {supported_geometries}, not '{geom_type}'"
            )  # pragma: no cover
        super().__init__(ScikitFemSubMesh3D)
        self.geom_type = geom_type
        self.gen_params = gen_params

    def _make_box_mesh(self, x_lim, y_lim, z_lim, h):
        """
        Create a structured box mesh.

        Parameters
        ----------
        x_lim, y_lim, z_lim : tuple
            Domain limits for each dimension
        h : float
            Target mesh size

        Returns
        -------
        skfem.MeshTet
            Box mesh with proper boundary tags
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
            "left": bnd_facets[np.isclose(midpoints[0], x_min)],
            "right": bnd_facets[np.isclose(midpoints[0], x_max)],
            "bottom": bnd_facets[np.isclose(midpoints[1], y_min)],
            "top": bnd_facets[np.isclose(midpoints[1], y_max)],
            "front": bnd_facets[np.isclose(midpoints[2], z_min)],
            "back": bnd_facets[np.isclose(midpoints[2], z_max)],
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
        height = z_lim[1] - z_lim[0]

        n_radial = max(3, int((r_outer - r_inner) / h) + 1)
        n_theta_base = max(12, int(2 * np.pi * r_outer / h))
        r_coords = np.linspace(r_inner, r_outer, n_radial)

        points_list = []
        for r in r_coords:
            if r > 1e-9:
                n_theta = max(6, int(n_theta_base * (r / r_outer)))
                thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
                for theta in thetas:
                    points_list.append([r * np.cos(theta), r * np.sin(theta)])

        points_2d = np.array(points_list)
        points_2d += 1e-8 * np.random.randn(*points_2d.shape)

        tri_2d = Delaunay(points_2d)
        triangles_base = tri_2d.simplices

        areas = []
        for tri in triangles_base:
            v = points_2d[tri]
            area = 0.5 * np.abs(
                (v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1])
                - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1])
            )
            areas.append(area)
        triangles_base = triangles_base[np.array(areas) > 1e-12]

        n_z = max(5, int(height / h))
        z_coords = np.linspace(z_lim[0], z_lim[1], n_z)
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

        tetrahedra = np.array(tetrahedra)
        volumes = []
        for tet in tetrahedra:
            v = nodes_3d[tet]
            vol = (
                np.abs(
                    np.linalg.det(
                        np.column_stack([v[1] - v[0], v[2] - v[0], v[3] - v[0]])
                    )
                )
                / 6
            )
            volumes.append(vol)
        tetrahedra = tetrahedra[np.array(volumes) > 1e-10]
        mesh = skfem.MeshTet(nodes_3d.T, np.array(tetrahedra).T)

        bottom_nodes = np.arange(n_nodes_per_layer)
        top_nodes = np.arange(nodes_3d.shape[0] - n_nodes_per_layer, nodes_3d.shape[0])

        points_2d_r = np.linalg.norm(points_2d, axis=1)
        inner_radius_nodes_2d = np.where(np.isclose(points_2d_r, r_inner, rtol=1e-3))[0]
        outer_radius_nodes_2d = np.where(np.isclose(points_2d_r, r_outer, rtol=1e-3))[0]

        inner_side_nodes = np.concatenate(
            [inner_radius_nodes_2d + (i * n_nodes_per_layer) for i in range(n_z)]
        )
        outer_side_nodes = np.concatenate(
            [outer_radius_nodes_2d + (i * n_nodes_per_layer) for i in range(n_z)]
        )

        boundary_facets = mesh.boundary_facets()
        facet_nodes = mesh.facets[:, boundary_facets]

        bottom_facets, top_facets, inner_facets, outer_facets = [], [], [], []

        for i, facet_idx in enumerate(boundary_facets):
            nodes_in_facet = facet_nodes[:, i]
            if np.all(np.isin(nodes_in_facet, bottom_nodes)):
                bottom_facets.append(facet_idx)
            elif np.all(np.isin(nodes_in_facet, top_nodes)):
                top_facets.append(facet_idx)
            elif np.any(np.isin(nodes_in_facet, inner_side_nodes)):
                inner_facets.append(facet_idx)
            elif np.any(np.isin(nodes_in_facet, outer_side_nodes)):
                outer_facets.append(facet_idx)

        boundaries = {}
        if bottom_facets:
            boundaries["bottom"] = np.array(bottom_facets)
        if top_facets:
            boundaries["top"] = np.array(top_facets)
        if inner_facets:
            boundaries["inner radius"] = np.array(inner_facets)
        if outer_facets:
            boundaries["outer radius"] = np.array(outer_facets)

        if boundaries:
            all_boundary_nodes = set()
            for facet_list in boundaries.values():
                all_boundary_nodes.update(mesh.facets[:, facet_list].flatten())
            mesh = laplacian_smooth(mesh, list(all_boundary_nodes), iterations=2)

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
        h = self.gen_params.get("h", 0.3)

        if self.geom_type == "box":
            x_key = next(k for k in lims if k.name == "x")
            y_key = next(k for k in lims if k.name == "y")
            z_key = next(k for k in lims if k.name == "z")
            x_lim = tuple(lims[x_key].values())
            y_lim = tuple(lims[y_key].values())
            z_lim = tuple(lims[z_key].values())
            coord_sys = "cartesian"
            mesh = self._make_box_mesh(x_lim, y_lim, z_lim, h)

        elif self.geom_type == "cylinder":
            r_key = next(k for k in lims if k.name == "r")
            z_key = next(k for k in lims if k.name == "z")
            r_lim = tuple(lims[r_key].values())
            z_lim = tuple(lims[z_key].values())
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
