import numpy as np
import pybamm
from .meshes import SubMesh
from meshpy.tet import MeshInfo, build
from meshpy.geometry import (
    make_box,
    make_cylinder,
)


def _num(val):
    if hasattr(val, "evaluate"):  # PyBaMM Scalar
        return float(val.evaluate())
    return float(val)


class SubMesh3D(SubMesh):
    """
    3D submesh class.
    Contains the position of the nodes, the number of mesh points, and
    (optionally) information about the tab locations.
    Parameters
    ----------
    edges_x : array_like
        An array containing the points corresponding to the edges of the submesh
        in x axis
    edges_y : array_like
        An array containing the points corresponding to the edges of the submesh
        in y axis
    edges_z : array_like
        An array containing the points corresponding to the edges of the submesh
        in z axis
    coord_sys : string
        The coordinate system of the submesh
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs
    """

    def __init__(self, edges_x, edges_y, edges_z, coord_sys, tabs=None):
        edges_x = np.array(edges_x)
        edges_y = np.array(edges_y)
        edges_z = np.array(edges_z)
        self.edges_x = edges_x
        self.edges_y = edges_y
        self.edges_z = edges_z

        self.tabs = tabs

        self.nodes_x = (edges_x[1:] + edges_x[:-1]) / 2
        self.nodes_y = (edges_y[1:] + edges_y[:-1]) / 2
        self.nodes_z = (edges_z[1:] + edges_z[:-1]) / 2

        self.d_edges_x = np.diff(self.edges_x)
        self.d_edges_y = np.diff(self.edges_y)
        self.d_edges_z = np.diff(self.edges_z)

        self.d_nodes_x = np.diff(self.nodes_x)
        self.d_nodes_y = np.diff(self.nodes_y)
        self.d_nodes_z = np.diff(self.nodes_z)

        X, Y, Z = np.meshgrid(self.nodes_x, self.nodes_y, self.nodes_z, indexing="ij")
        self.nodes = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        self.npts_x = self.nodes_x.size
        self.npts_y = self.nodes_y.size
        self.npts_z = self.nodes_z.size
        self.npts = self.npts_x * self.npts_y * self.npts_z
        self.coord_sys = coord_sys
        self.dimension = 3
        self.internal_boundaries = []

    @staticmethod
    def read_lims(lims):
        """
        Expects lims to be a dict with exactly keys
          { x_var: {min, max, …}, y_var: {…}, z_var: {…}, tabs? }
        Pops off "tabs" if present and returns:
          (x_var, x_lims, y_var, y_lims, z_var, z_lims, tabs)
        """
        tabs = lims.pop("tabs", None)

        if len(lims) != 3:
            raise pybamm.GeometryError(
                "3D lims must have exactly three spatial variables"
            )

        (var_x, lims_x), (var_y, lims_y), (var_z, lims_z) = lims.items()

        if isinstance(var_x, str):
            var_x = getattr(pybamm.standard_spatial_vars, var_x)
        if isinstance(var_y, str):
            var_y = getattr(pybamm.standard_spatial_vars, var_y)
        if isinstance(var_z, str):
            var_z = getattr(pybamm.standard_spatial_vars, var_z)

        return var_x, lims_x, var_y, lims_y, var_z, lims_z, tabs

    def to_json(self):
        d = {
            "edges_x": self.edges_x.tolist(),
            "edges_y": self.edges_y.tolist(),
            "edges_z": self.edges_z.tolist(),
            "coord_sys": self.coord_sys,
        }
        if self.tabs is not None:
            d["tabs"] = self.tabs
        return d

    @classmethod
    def _from_json(cls, json_dict):
        return cls(
            json_dict["edges_x"],
            json_dict["edges_y"],
            json_dict["edges_z"],
            json_dict["coord_sys"],
            tabs=json_dict.get("tabs"),
        )

    def create_ghost_cell(self, side):
        if side == "left":
            gx = np.array([2 * self.edges_x[0] - self.edges_x[1], self.edges_x[0]])
            gy, gz = self.edges_y, self.edges_z
        elif side == "right":
            gx = np.array([self.edges_x[-1], 2 * self.edges_x[-1] - self.edges_x[-2]])
            gy, gz = self.edges_y, self.edges_z
        elif side == "front":
            gy = np.array([2 * self.edges_y[0] - self.edges_y[1], self.edges_y[0]])
            gx, gz = self.edges_x, self.edges_z
        elif side == "back":
            gy = np.array([self.edges_y[-1], 2 * self.edges_y[-1] - self.edges_y[-2]])
            gx, gz = self.edges_x, self.edges_z
        elif side == "bottom":
            gz = np.array([2 * self.edges_z[0] - self.edges_z[1], self.edges_z[0]])
            gx, gy = self.edges_x, self.edges_y
        elif side == "top":
            gz = np.array([self.edges_z[-1], 2 * self.edges_z[-1] - self.edges_z[-2]])
            gx, gy = self.edges_x, self.edges_y
        else:
            raise ValueError(f"Invalid side: {side}")

        return SubMesh3D(gx, gy, gz, self.coord_sys, tabs=self.tabs)


class Uniform3DSubMesh(SubMesh3D):
    """
    A 3D submesh with uniform spacing in all dimensions
    """

    @staticmethod
    def read_lims(lims):
        """
        Parser for uniform spacing 3D submesh.
        Exactly the same logic as in SubMesh3D.read_lims,
        but re-defined here so that Uniform3DSubMesh.read_lims
        is always the one called by its __init__.
        """
        tabs = lims.pop("tabs", None)
        if len(lims) != 3:
            raise pybamm.GeometryError(
                "3D lims must have exactly three spatial variables"
            )
        (var1, lims1), (var2, lims2), (var3, lims3) = lims.items()

        if isinstance(var1, str):
            var1 = getattr(pybamm.standard_spatial_vars, var1)
        if isinstance(var2, str):
            var2 = getattr(pybamm.standard_spatial_vars, var2)
        if isinstance(var3, str):
            var3 = getattr(pybamm.standard_spatial_vars, var3)

        return var1, lims1, var2, lims2, var3, lims3, tabs

    def __init__(self, lims, npts):
        var_x, lims_x, var_y, lims_y, var_z, lims_z, tabs = self.read_lims(lims)
        nx, ny, nz = npts[var_x.name], npts[var_y.name], npts[var_z.name]

        edges_x = np.linspace(_num(lims_x["min"]), _num(lims_x["max"]), nx + 1)
        edges_y = np.linspace(_num(lims_y["min"]), _num(lims_y["max"]), ny + 1)
        edges_z = np.linspace(_num(lims_z["min"]), _num(lims_z["max"]), nz + 1)

        coord_sys = var_x.coord_sys
        if var_y.coord_sys != coord_sys or var_z.coord_sys != coord_sys:
            raise pybamm.GeometryError("Coord systems must agree for 3D mesh")

        super().__init__(edges_x, edges_y, edges_z, coord_sys, tabs=tabs)


class MeshPyGenerator3D:
    """
    A MeshGenerator-style wrapper that calls meshpy under the hood.
    Usage: Generator(geometry, params) returns a SubMesh-like object
    with attributes .nodes, .elements, .npts, .dimension=3, etc.
    """

    def __init__(self, geom_type, **gen_params):
        self.geom_type = geom_type
        self.params = gen_params

    def _make_spiral_jelly_roll(
        inner_radius, outer_radius, height, turns, max_volume=None
    ):
        """
        Generate a proper spiral jelly roll mesh with Archimedean spiral cross-section.

        Parameters:
        -----------
        inner_radius : float
            Inner radius of the spiral
        outer_radius : float
            Outer radius of the spiral
        height : float
            Height of the jelly roll
        turns : float
            Number of spiral turns
        max_volume : float, optional
            Maximum volume constraint for mesh elements

        Returns:
        --------
        MeshInfo object ready for tetrahedral mesh generation
        """
        # Generate spiral points using Archimedean spiral: r = a + b*theta
        n_theta = int(200 * turns)  # Points per turn
        theta_max = 2 * np.pi * turns
        theta = np.linspace(0, theta_max, n_theta)

        # Archimedean spiral parameters
        a = inner_radius
        b = (outer_radius - inner_radius) / theta_max
        r_spiral = a + b * theta

        spiral_x = r_spiral * np.cos(theta)
        spiral_y = r_spiral * np.sin(theta)

        n_z = max(10, int(height * 20))
        z_levels = np.linspace(0, height, n_z)

        points = []

        for z in z_levels:
            for x, y in zip(spiral_x, spiral_y):
                points.append((x, y, z))

        for z in z_levels:
            points.append((0, 0, z))

        n_boundary = 50
        boundary_theta = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
        for z in z_levels:
            for t in boundary_theta:
                x = outer_radius * np.cos(t)
                y = outer_radius * np.sin(t)
                points.append((x, y, z))

        facets = []

        bottom_center_idx = len(spiral_x) * n_z
        for i in range(len(spiral_x) - 1):
            facets.append([bottom_center_idx, i, i + 1])

        top_center_idx = (
            bottom_center_idx + n_z - 1
        )  # Index of center point at z=height
        top_offset = (n_z - 1) * len(spiral_x)
        for i in range(len(spiral_x) - 1):
            facets.append([top_center_idx, top_offset + i + 1, top_offset + i])

        for k in range(n_z - 1):
            for i in range(len(spiral_x) - 1):
                curr_i = k * len(spiral_x) + i
                curr_i1 = k * len(spiral_x) + i + 1

                next_i = (k + 1) * len(spiral_x) + i
                next_i1 = (k + 1) * len(spiral_x) + i + 1

                facets.append([curr_i, next_i, curr_i1])
                facets.append([curr_i1, next_i, next_i1])

        mesh_info = MeshInfo()
        mesh_info.set_points(points)
        mesh_info.set_facets(facets)

        return mesh_info

    def _make_cylinder_simple(radius, height, max_volume=None):
        """
        Simple cylinder mesh generation using meshpy's built-in function.
        """
        cylinder_pts, cylinder_facets, cylinder_holes, cylinder_fm = make_cylinder(
            radius=radius, height=height
        )

        mesh_info = MeshInfo()
        mesh_info.set_points(cylinder_pts)
        mesh_info.set_facets(cylinder_facets)

        if cylinder_holes:
            mesh_info.set_holes(cylinder_holes)

        return mesh_info

    def __call__(self, lims, npts):
        # Convert SpatialVariable keys to string keys
        lims_dict = {}
        for k, v in lims.items():
            key_name = k if isinstance(k, str) else k.name
            lims_dict[key_name] = v

        if self.geom_type == "box":
            origin = (
                _num(lims_dict["x"]["min"]),
                _num(lims_dict["y"]["min"]),
                _num(lims_dict["z"]["min"]),
            )
            lengths = (
                _num(lims_dict["x"]["max"]) - origin[0],
                _num(lims_dict["y"]["max"]) - origin[1],
                _num(lims_dict["z"]["max"]) - origin[2],
            )
            box_pts, box_facets, box_holes, box_fm = make_box(origin, lengths)
            mesh_info = MeshInfo()
            mesh_info.set_points(box_pts)
            if box_holes:
                mesh_info.set_holes(box_holes)
            if box_fm:
                mesh_info.set_facets(box_facets, markers=box_fm)
            else:
                mesh_info.set_facets(box_facets)

        elif self.geom_type == "cylinder":
            mesh_info = MeshPyGenerator3D._make_cylinder_simple(
                self.params.get("radius", 1.0),
                self.params.get("height", 1.0),
                self.params.get("max_volume"),
            )

        elif self.geom_type == "spiral":
            mesh_info = MeshPyGenerator3D._make_spiral_jelly_roll(
                self.params.get("inner_radius", 0.1),
                self.params.get("outer_radius", 1.0),
                self.params.get("height", 1.0),
                self.params.get("turns", 3.0),
                self.params.get("max_volume"),
            )

        else:
            raise ValueError(f"Unknown geom_type: {self.geom_type}")

        mesh = build(mesh_info, max_volume=self.params.get("max_volume", 1e-2))

        sub = SubMesh3D.__new__(SubMesh3D)
        sub.nodes = np.array(mesh.points)
        sub.elements = np.array(mesh.elements)
        sub.npts = sub.nodes.shape[0]
        sub.dimension = 3
        return sub
