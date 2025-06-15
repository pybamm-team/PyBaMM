import numpy as np

import pybamm
from pybamm.util import import_optional_dependency


def _num(val):
    if hasattr(val, "evaluate"):  # PyBaMM Scalar
        return float(val.evaluate())
    return float(val)


class ScikitFemGenerator3D(pybamm.MeshGenerator):
    """
    A MeshGenerator that uses scikit-fem for 3D mesh generation.
    """

    def __init__(self, geom_type, **gen_params):
        super().__init__(ScikitFemSubMesh3D)
        self.geom_type = geom_type
        self.gen_params = gen_params

    def _create_mesh_from_points(self, points):
        """Helper to create a mesh from a point cloud."""
        skfem = import_optional_dependency("skfem")
        from scipy.spatial import Delaunay

        unique_points, inverse_indices = np.unique(points, axis=0, return_inverse=True)

        if unique_points.shape[0] < 4:
            pybamm.logger.warning("Mesh has too few unique points for 3D Delaunay.")
            return None

        try:
            delaunay = Delaunay(unique_points)
            mesh = skfem.MeshTet(unique_points.T, delaunay.simplices.T)

            subdomains = {"default": np.arange(mesh.nelements)}
            mesh = mesh.with_subdomains(subdomains)

            return mesh
        except Exception as e:
            pybamm.logger.warning(f"Delaunay triangulation failed: {e}")
            return None

    def _make_box_mesh(self, x_lim, y_lim, z_lim, h):
        skfem = import_optional_dependency("skfem")
        x_min, x_max = x_lim
        y_min, y_max = y_lim
        z_min, z_max = z_lim

        nx = max(5, int((x_max - x_min) / h))
        ny = max(5, int((y_max - y_min) / h))
        nz = max(5, int((z_max - z_min) / h))

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

        mesh = mesh.with_boundaries(boundaries).with_subdomains(subdomains)
        return mesh

    def _make_cylinder_mesh(self, radius, height, h):
        n_radial = max(5, int(radius / h))
        n_theta = max(12, int(2 * np.pi * radius / h))
        n_z = max(5, int(height / h))

        r_coords = np.linspace(0, radius, n_radial)
        theta_coords = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        z_coords = np.linspace(0, height, n_z)

        points = []
        for z in z_coords:
            for r in r_coords:
                if r == 0:
                    points.append([0, 0, z])
                else:
                    for theta in theta_coords:
                        points.append([r * np.cos(theta), r * np.sin(theta), z])

        points = np.array(points)

        np.random.seed(0)  # for reproducibility
        jitter = h * 1e-5
        points += np.random.normal(scale=jitter, size=points.shape)

        mesh = self._create_mesh_from_points(points)  # Pass jittered points

        if mesh is None:
            return None

        bnd_facets = mesh.boundary_facets()
        if len(bnd_facets) == 0:
            return None

        midpoints = mesh.p[:, mesh.facets[:, bnd_facets]].mean(axis=1)

        # Use tolerances for boundary detection
        tol = h * 0.1
        boundaries = {}

        bottom_mask = midpoints[2] < (0 + tol)
        if np.any(bottom_mask):
            boundaries["bottom cap"] = bnd_facets[bottom_mask]

        top_mask = midpoints[2] > (height - tol)
        if np.any(top_mask):
            boundaries["top cap"] = bnd_facets[top_mask]

        radial_dist = np.sqrt(midpoints[0] ** 2 + midpoints[1] ** 2)
        side_mask = radial_dist > (radius - tol)
        if np.any(side_mask):
            boundaries["side wall"] = bnd_facets[side_mask]

        if not boundaries:
            return None

        subdomains = {"default": np.arange(mesh.nelements)}
        mesh = mesh.with_boundaries(boundaries).with_subdomains(subdomains)

        return mesh

    def _make_spiral_mesh(self, inner_radius, outer_radius, height, turns, h):
        n_radial = max(4, int((outer_radius - inner_radius) / h))
        n_z = max(4, int(height / h))
        n_theta_total = max(20, int(turns * 2 * np.pi / (h / inner_radius)))

        points = []
        for z_val in np.linspace(0, height, n_z):
            for r_val in np.linspace(inner_radius, outer_radius, n_radial):
                num_theta_for_layer = max(
                    8, int(n_theta_total * (r_val / outer_radius))
                )
                for theta_val in np.linspace(0, turns * 2 * np.pi, num_theta_for_layer):
                    points.append(
                        [r_val * np.cos(theta_val), r_val * np.sin(theta_val), z_val]
                    )

        mesh = self._create_mesh_from_points(np.array(points))
        if mesh is None:
            return None

        bnd_facets = mesh.boundary_facets()
        midpoints = mesh.p[:, mesh.facets[:, bnd_facets]].mean(axis=1)
        boundaries = {
            "bottom": bnd_facets[np.isclose(midpoints[2], 0)],
            "top": bnd_facets[np.isclose(midpoints[2], height)],
            "inner wall": bnd_facets[
                np.isclose(np.sqrt(midpoints[0] ** 2 + midpoints[1] ** 2), inner_radius)
            ],
            "outer wall": bnd_facets[
                np.isclose(np.sqrt(midpoints[0] ** 2 + midpoints[1] ** 2), outer_radius)
            ],
        }

        subdomains = {"default": np.arange(mesh.nelements)}
        mesh = mesh.with_boundaries(boundaries).with_subdomains(subdomains)

        return mesh

    def __call__(self, lims, npts):
        """Main entry point called by PyBaMM's Discretisation."""
        h = self.gen_params.get("h", 0.3)

        if self.geom_type == "box":
            x_key = next(k for k in lims if k.name == "x")
            y_key = next(k for k in lims if k.name == "y")
            z_key = next(k for k in lims if k.name == "z")
            x_lim = tuple(lims[x_key].values())
            y_lim = tuple(lims[y_key].values())
            z_lim = tuple(lims[z_key].values())
            mesh = self._make_box_mesh(x_lim, y_lim, z_lim, h)
        elif self.geom_type == "cylinder":
            # --- FIX: Pass the correct arguments ---
            radius = self.gen_params.get("radius", 0.4)
            height = self.gen_params.get("height", 0.8)
            mesh = self._make_cylinder_mesh(radius, height, h)
        elif self.geom_type == "spiral":
            # --- FIX: Pass the correct arguments ---
            inner_radius = self.gen_params.get("inner_radius", 0.1)
            outer_radius = self.gen_params.get("outer_radius", 0.4)
            height = self.gen_params.get("height", 0.8)
            turns = self.gen_params.get("turns", 2.0)
            mesh = self._make_spiral_mesh(inner_radius, outer_radius, height, turns, h)
        else:
            raise ValueError(f"Unknown geom_type: {self.geom_type}")

        if mesh is None:
            raise pybamm.DiscretisationError(
                f"Mesh generation failed for {self.geom_type}"
            )

        nodes = mesh.p.T
        elements = mesh.t.T
        submesh = self.submesh_type(mesh, nodes, elements, "cartesian")
        return submesh


class ScikitFemSubMesh3D(pybamm.SubMesh):
    """
    A 3D submesh class for unstructured meshes generated by scikit-fem.
    This class wraps scikit-fem's mesh objects while maintaining compatibility
    with PyBaMM's mesh system.

    Parameters
    ----------
    skfem_mesh : skfem.MeshTet
        The scikit-fem mesh object
    nodes : array_like
        Array of node coordinates (npts, 3)
    elements : array_like
        Array of element connectivity (nelements, 4)
    coord_sys : string
        The coordinate system of the submesh
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs
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
        self.edges = self._skfem_mesh.edges.T

        self.basis = skfem.InteriorBasis(self._skfem_mesh, skfem.ElementTetP1())

        self.facet_basis = skfem.FacetBasis(self._skfem_mesh, self.basis.elem)

        if hasattr(self._skfem_mesh, "boundaries"):
            for name in self._skfem_mesh.boundaries:
                facet_basis = skfem.FacetBasis(
                    self._skfem_mesh,
                    self.basis.elem,
                    facets=self._skfem_mesh.boundaries[name],
                )
                setattr(self, f"{name}_basis", facet_basis)

                dofs = self.basis.get_dofs(
                    name
                ).all()  # Use .all() to get the numpy array
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

    def _compute_boundary_faces(self):
        """Compute boundary faces and their normals for FVM."""
        faces = []
        for elem in self.elements:
            faces.extend(
                [
                    tuple(sorted([elem[0], elem[1], elem[2]])),
                    tuple(sorted([elem[0], elem[1], elem[3]])),
                    tuple(sorted([elem[0], elem[2], elem[3]])),
                    tuple(sorted([elem[1], elem[2], elem[3]])),
                ]
            )

        from collections import Counter

        face_counts = Counter(faces)

        self.boundary_faces = [
            face for face, count in face_counts.items() if count == 1
        ]

        self.boundary_normals = []
        self.boundary_centroids = []

        for face in self.boundary_faces:
            face_indices = list(face)
            v1 = self.nodes[face_indices[0]]
            v2 = self.nodes[face_indices[1]]
            v3 = self.nodes[face_indices[2]]

            # Compute face centroid
            centroid = (v1 + v2 + v3) / 3
            self.boundary_centroids.append(centroid)

            # Compute face normal using cross product
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)  # Normalize
            self.boundary_normals.append(normal)

        self.boundary_normals = np.array(self.boundary_normals)
        self.boundary_centroids = np.array(self.boundary_centroids)

    def to_json(self):
        """Convert mesh to JSON format."""
        d = {
            "mesh_type": self.__class__.__name__,
            "nodes": self.nodes.tolist(),
            "elements": self.elements.tolist(),
            "coord_sys": self.coord_sys,
            "element_volumes": self.element_volumes.tolist(),
            "element_centroids": self.element_centroids.tolist(),
            "boundary_faces": [list(f) for f in self.boundary_faces],
            "boundary_normals": self.boundary_normals.tolist(),
            "boundary_centroids": self.boundary_centroids.tolist(),
        }
        if self.tabs is not None:
            d["tabs"] = self.tabs
        return d

    @classmethod
    def _from_json(cls, json_dict):
        """Create mesh from JSON format."""
        skfem = import_optional_dependency("skfem")

        nodes = np.array(json_dict["nodes"])
        elements = np.array(json_dict["elements"])
        mesh = skfem.MeshTet(nodes.T, elements.T)

        submesh = cls(
            mesh, nodes, elements, json_dict["coord_sys"], tabs=json_dict.get("tabs")
        )

        submesh.element_volumes = np.array(json_dict["element_volumes"])
        submesh.element_centroids = np.array(json_dict["element_centroids"])
        submesh.boundary_faces = [tuple(f) for f in json_dict["boundary_faces"]]
        submesh.boundary_normals = np.array(json_dict["boundary_normals"])
        submesh.boundary_centroids = np.array(json_dict["boundary_centroids"])

        return submesh
