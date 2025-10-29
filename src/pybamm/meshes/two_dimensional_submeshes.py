#
# Two-dimensional submeshes
#
import numpy as np

import pybamm

from .meshes import SubMesh


class SubMesh2D(SubMesh):
    """
    2D submesh class.
    Contains the position of the nodes, the number of mesh points, and
    (optionally) information about the tab locations.

    Parameters
    ----------
    edges_lr : array_like
        An array containing the points corresponding to the edges of the submesh
        in the left-right direction
    edges_tb : array_like
        An array containing the points corresponding to the edges of the submesh
        in the top-bottom direction
    coord_sys_lr : string
        The coordinate system of the submesh in the left-right direction
    coord_sys_tb : string
        The coordinate system of the submesh in the top-bottom direction
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs
    """

    def __init__(self, edges_lr, edges_tb, coord_sys, tabs=None):
        self.edges_lr = edges_lr
        self.edges_tb = edges_tb
        self.nodes_lr = (self.edges_lr[1:] + self.edges_lr[:-1]) / 2
        self.nodes_tb = (self.edges_tb[1:] + self.edges_tb[:-1]) / 2
        self.d_edges_lr = np.diff(self.edges_lr)
        self.d_edges_tb = np.diff(self.edges_tb)
        self.d_nodes_lr = np.diff(self.nodes_lr)
        self.d_nodes_tb = np.diff(self.nodes_tb)
        self.npts_lr = self.nodes_lr.size
        self.npts_tb = self.nodes_tb.size
        self.npts = self.npts_lr * self.npts_tb
        self.coord_sys = coord_sys
        self.internal_boundaries = []
        self.dimension = 2

        # Add tab locations in terms of "left" and "right"
        self.tabs = tabs

    def read_lims(self, lims):
        # Separate limits and tabs
        # Read and remove tabs. If "tabs" is not a key in "lims", then tabs is set to
        # "None" and nothing is removed from lims
        tabs = lims.pop("tabs", None)

        # check that only one variable passed in
        if len(lims) != 2:
            raise pybamm.GeometryError("lims should only contain two variables")

        ((spatial_var_lr, spatial_lims_lr), (spatial_var_tb, spatial_lims_tb)) = (
            lims.items()
        )

        if isinstance(spatial_var_lr, str):
            spatial_var_lr = getattr(pybamm.standard_spatial_vars, spatial_var_lr)
        if isinstance(spatial_var_tb, str):
            spatial_var_tb = getattr(pybamm.standard_spatial_vars, spatial_var_tb)

        return spatial_var_lr, spatial_lims_lr, spatial_var_tb, spatial_lims_tb, tabs

    def to_json(self):
        json_dict = {
            "edges_lr": self.edges_lr.tolist(),
            "edges_tb": self.edges_tb.tolist(),
            "coord_sys": self.coord_sys,
        }

        if hasattr(self, "tabs"):
            json_dict["tabs"] = self.tabs

        return json_dict

    def create_ghost_cell(self, side):
        if side == "left":
            gs_edges_lr = np.array(
                [2 * self.edges_lr[0] - self.edges_lr[1], self.edges_lr[0]]
            )
            gs_edges_tb = self.edges_tb
        elif side == "right":
            gs_edges_lr = np.array(
                [self.edges_lr[-1], 2 * self.edges_lr[-1] - self.edges_lr[-2]]
            )
            gs_edges_tb = self.edges_tb
        elif side == "top":
            gs_edges_lr = self.edges_lr
            gs_edges_tb = np.array(
                [2 * self.edges_tb[0] - self.edges_tb[1], self.edges_tb[0]]
            )
        elif side == "bottom":
            gs_edges_lr = self.edges_lr
            gs_edges_tb = np.array(
                [self.edges_tb[-1], 2 * self.edges_tb[-1] - self.edges_tb[-2]]
            )
        else:
            raise ValueError(f"Invalid side: {side}")
        gs_submesh = pybamm.SubMesh2D(gs_edges_lr, gs_edges_tb, self.coord_sys)
        return gs_submesh


class Uniform2DSubMesh(SubMesh2D):
    """
    A 2D submesh with uniform spacing in both dimensions
    """

    def __init__(self, lims, npts):
        spatial_var_lr, spatial_lims_lr, spatial_var_tb, spatial_lims_tb, tabs = (
            self.read_lims(lims)
        )
        npts_lr = npts[spatial_var_lr.name]
        npts_tb = npts[spatial_var_tb.name]

        edges_lr = np.linspace(
            spatial_lims_lr["min"], spatial_lims_lr["max"], npts_lr + 1
        )
        edges_tb = np.linspace(
            spatial_lims_tb["min"], spatial_lims_tb["max"], npts_tb + 1
        )
        if spatial_var_lr.coord_sys != spatial_var_tb.coord_sys:
            raise pybamm.GeometryError(
                "Coordinate systems must be the same for 2D submeshes"
            )
        coord_sys = spatial_var_lr.coord_sys

        super().__init__(edges_lr, edges_tb, coord_sys, tabs=tabs)

    def read_lims(self, lims):
        # Separate limits and tabs
        # Read and remove tabs. If "tabs" is not a key in "lims", then tabs is set to
        # "None" and nothing is removed from lims
        tabs = lims.pop("tabs", None)

        # check that only one variable passed in
        if len(lims) != 2:
            raise pybamm.GeometryError("lims should only contain two variables")

        ((spatial_var1, spatial_lims1), (spatial_var2, spatial_lims2)) = lims.items()

        if isinstance(spatial_var1, str):
            spatial_var1 = getattr(pybamm.standard_spatial_vars, spatial_var1)
        if isinstance(spatial_var2, str):
            spatial_var2 = getattr(pybamm.standard_spatial_vars, spatial_var2)

        return spatial_var1, spatial_lims1, spatial_var2, spatial_lims2, tabs
