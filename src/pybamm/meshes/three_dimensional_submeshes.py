import numpy as np
from meshio import Mesh
from pybamm.meshes.meshes import SubMesh
import pygmsh


class SubMesh3D(SubMesh):
    """Unstructured 3D submesh from pygmsh + meshio."""

    def __init__(self, mesh: Mesh):
        self.points = mesh.points
        self.cells = {c.type: c.data for c in mesh.cells}
        self.tetrahedra = self.cells.get("tetra", np.empty((0, 4), int))
        self.triangles = self.cells.get("triangle", None)

        self._compute_volumes_and_centres()

        self._build_adjacency()
        self.coord_sys = "cartesian"
        self.internal_boundaries = []

    def _compute_volumes_and_centres(self):
        pts = self.points
        tets = self.tetrahedra

        v1 = pts[tets[:, 1]] - pts[tets[:, 0]]
        v2 = pts[tets[:, 2]] - pts[tets[:, 0]]
        v3 = pts[tets[:, 3]] - pts[tets[:, 0]]
        self.volumes = np.abs(np.einsum("ij,ij->i", v1, np.cross(v2, v3))) / 6

        self.nodes = pts[tets].mean(axis=1)
        self.npts = len(self.volumes)

    def _build_adjacency(self):
        faces = {}
        for ci, tet in enumerate(self.tetrahedra):
            for face in [
                (tet[a], tet[b], tet[c])
                for a, b, c in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
            ]:
                key = tuple(sorted(face))
                faces.setdefault(key, []).append(ci)
        self.adjacency_list = [[] for _ in range(self.npts)]
        for cell_pair in faces.values():
            if len(cell_pair) == 2:
                a, b = cell_pair
                self.adjacency_list[a].append(b)
                self.adjacency_list[b].append(a)
        return self


class PyGmshMeshGenerator:
    """Generate various 3D shapes via pygmsh, return SubMesh3D."""

    def __init__(self, mesh_size=0.1):
        self.mesh_size = mesh_size

    def generate(self, geom_type: str, params: dict) -> SubMesh3D:
        """Dispatch to shape-specific generator."""
        with pygmsh.geo.Geometry() as geom:
            if geom_type == "rectangular":
                self._add_rectangular(geom, params)
            elif geom_type == "cylindrical":
                self._add_cylindrical(geom, params)
            elif geom_type == "spiral":
                self._add_spiral(geom, params)
            else:
                raise ValueError(f"Unknown 3D geometry '{geom_type}'")
            mesh = geom.generate_mesh()
        meshio_mesh = Mesh(
            points=mesh.points, cells={c.type: c.data for c in mesh.cells}
        )
        return SubMesh3D(meshio_mesh)

    def _add_rectangular(self, geom, p):
        geom.add_box(
            x0=p["x"][0],
            y0=p["y"][0],
            z0=p["z"][0],
            x1=p["x"][1],
            y1=p["y"][1],
            z1=p["z"][1],
            mesh_size=self.mesh_size,
        )

    def _add_cylindrical(self, geom, p):
        origin = [0, 0, 0]
        axis = [0, 0, p["height"]]
        geom.add_cylinder(origin, axis, p["radius"], mesh_size=self.mesh_size)

    def _add_spiral(self, geom, p):
        coil = geom.add_parametric_surface(
            lambda u, v: (
                (p["inner_radius"] + (p["outer_radius"] - p["inner_radius"]) * u)
                * np.cos(2 * np.pi * p["turns"] * v),
                (p["inner_radius"] + (p["outer_radius"] - p["inner_radius"]) * u)
                * np.sin(2 * np.pi * p["turns"] * v),
                p["height"] * v,
            ),
            [0, 1],
            [0, 1],
            mesh_size=self.mesh_size,
        )
        geom.add_surface_loop([coil])
