import numpy as np
import pybamm
from .meshes import SubMesh
from pybamm.util import import_optional_dependency


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
        x_flat_f = X.flatten(order="F")
        y_flat_f = Y.flatten(order="F")
        z_flat_f = Z.flatten(order="F")
        self.nodes = np.column_stack([x_flat_f, y_flat_f, z_flat_f])

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


class ScikitFemGenerator3D:
    """
    A MeshGenerator-style wrapper that uses scikit-fem for 3D mesh generation.
    Usage: Generator(geometry, params) returns a SubMesh-like object
    with attributes .nodes, .elements, .npts, .dimension=3, etc.
    """

    def __init__(self, geom_type, **gen_params):
        self.geom_type = geom_type
        self.params = gen_params

    def _make_box_mesh(self, x_min, x_max, y_min, y_max, z_min, z_max, h=None):
        """
        Generate a 3D box mesh using scikit-fem's built-in functionality.

        Parameters:
        -----------
        x_min, x_max, y_min, y_max, z_min, z_max : float
            Bounds of the box
        h : float, optional
            Mesh size parameter (smaller = finer mesh)
        """
        skfem = import_optional_dependency("skfem")
        if h is None:
            h = min(x_max - x_min, y_max - y_min, z_max - z_min) / 10

        # Create a structured tetrahedral mesh for a box
        # scikit-fem provides MeshTet.init_tensor for structured 3D meshes
        nx = max(5, int((x_max - x_min) / h))
        ny = max(5, int((y_max - y_min) / h))
        nz = max(5, int((z_max - z_min) / h))

        mesh = skfem.MeshTet.init_tensor(
            np.linspace(x_min, x_max, nx),
            np.linspace(y_min, y_max, ny),
            np.linspace(z_min, z_max, nz),
        )

        return mesh

    def _make_cylinder_mesh(self, radius, height, h=None):
        """
        Generate a cylindrical mesh using scikit-fem.
        This creates a structured mesh by revolution or extrusion.

        Parameters:
        -----------
        radius : float
            Cylinder radius
        height : float
            Cylinder height
        h : float, optional
            Mesh size parameter
        """
        skfem = import_optional_dependency("skfem")
        if h is None:
            h = min(radius, height) / 10

        # Create cylinder by extruding a circular cross-section
        # First create a 2D circular mesh, then extrude in z-direction
        n_radial = max(5, int(radius / h))
        n_theta = max(12, int(2 * np.pi * radius / h))
        n_z = max(5, int(height / h))

        r = np.linspace(0, radius, n_radial)
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        z = np.linspace(0, height, n_z)

        points = []
        for z_val in z:
            for r_val in r:
                if r_val == 0:  # Center point
                    points.append([0, 0, z_val])
                else:
                    for theta_val in theta:
                        x = r_val * np.cos(theta_val)
                        y = r_val * np.sin(theta_val)
                        points.append([x, y, z_val])

        points = np.array(points)

        try:
            from scipy.spatial import Delaunay

            tri = Delaunay(points)
            mesh = skfem.MeshTet(points.T, tri.simplices.T)
        except ImportError:
            mesh = self._make_box_mesh(-radius, radius, -radius, radius, 0, height, h)

        return mesh

    def _make_spiral_mesh(self, inner_radius, outer_radius, height, turns, h=None):
        """
        Generate a spiral jelly roll mesh using scikit-fem.
        Creates an Archimedean spiral cross-section and extrudes it.

        Parameters:
        -----------
        inner_radius : float
            Inner radius of the spiral
        outer_radius : float
            Outer radius of the spiral
        height : float
            Height of the spiral
        turns : float
            Number of spiral turns
        h : float, optional
            Mesh size parameter
        """
        skfem = import_optional_dependency("skfem")
        if h is None:
            h = min(outer_radius - inner_radius, height) / 20

        # Generate spiral geometry
        n_theta = int(100 * turns)
        n_z = max(10, int(height / h))

        theta_max = 2 * np.pi * turns
        theta = np.linspace(0, theta_max, n_theta)
        z_levels = np.linspace(0, height, n_z)

        # Archimedean spiral: r = a + b*theta
        a = inner_radius
        b = (outer_radius - inner_radius) / theta_max
        r_spiral = a + b * theta

        points = []

        # Generate spiral points at each z-level
        for z_val in z_levels:
            # Center point
            points.append([0, 0, z_val])

            # Spiral points
            for _i, (r_val, theta_val) in enumerate(zip(r_spiral, theta)):
                x = r_val * np.cos(theta_val)
                y = r_val * np.sin(theta_val)
                points.append([x, y, z_val])

        points = np.array(points)

        # Create tetrahedral mesh
        try:
            from scipy.spatial import Delaunay

            tri = Delaunay(points)
            mesh = skfem.MeshTet(points.T, tri.simplices.T)
        except ImportError:
            # Fallback: create cylindrical approximation
            mesh = self._make_cylinder_mesh(outer_radius, height, h)

        return mesh

    def __call__(self, lims, npts):
        """
        Generate 3D mesh based on geometry type and parameters.

        Parameters:
        -----------
        lims : dict
            Dictionary containing spatial limits
        npts : dict
            Dictionary containing number of points (may be ignored for unstructured meshes)

        Returns:
        --------
        SubMesh3D-like object with nodes, elements, npts, dimension attributes
        """
        # Convert SpatialVariable keys to string keys
        lims_dict = {}
        for k, v in lims.items():
            key_name = k if isinstance(k, str) else k.name
            lims_dict[key_name] = v

        # Extract mesh size parameter
        h = self.params.get("h", self.params.get("max_volume", None))
        if h is not None and h > 1:
            # Convert volume constraint to approximate edge length
            h = h ** (1 / 3)

        if self.geom_type == "box":
            x_min = _num(lims_dict["x"]["min"])
            x_max = _num(lims_dict["x"]["max"])
            y_min = _num(lims_dict["y"]["min"])
            y_max = _num(lims_dict["y"]["max"])
            z_min = _num(lims_dict["z"]["min"])
            z_max = _num(lims_dict["z"]["max"])

            mesh = self._make_box_mesh(x_min, x_max, y_min, y_max, z_min, z_max, h)

        elif self.geom_type == "cylinder":
            radius = self.params.get("radius", 1.0)
            height = self.params.get("height", 1.0)
            mesh = self._make_cylinder_mesh(radius, height, h)

        elif self.geom_type == "spiral":
            inner_radius = self.params.get("inner_radius", 0.1)
            outer_radius = self.params.get("outer_radius", 1.0)
            height = self.params.get("height", 1.0)
            turns = self.params.get("turns", 3.0)
            mesh = self._make_spiral_mesh(inner_radius, outer_radius, height, turns, h)

        else:
            raise ValueError(f"Unknown geom_type: {self.geom_type}")

        sub = SubMesh3D.__new__(SubMesh3D)
        sub.nodes = (
            mesh.p.T
        )  # scikit-fem stores points as (dim, npts), we want (npts, dim)
        sub.elements = mesh.t.T
        sub.npts = sub.nodes.shape[0]
        sub.dimension = 3
        if self.geom_type == "cylinder":
            sub.coord_sys = "cylindrical polar"
        elif self.geom_type == "spiral":
            sub.coord_sys = "spiral"
        else:
            sub.coord_sys = "cartesian"
        sub.internal_boundaries = []

        sub._skfem_mesh = mesh

        return sub
