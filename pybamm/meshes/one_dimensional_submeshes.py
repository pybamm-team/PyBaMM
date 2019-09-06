#
# One-dimensional submeshes
#
import pybamm

import numpy as np


class SubMesh1D:
    """
    1D submesh class.
    Contains the position of the nodes, the number of mesh points, and
    (optionally) information about the tab locations.
    """

    def __init__(self, edges, coord_sys, tabs=None):
        self.edges = edges
        self.nodes = (self.edges[1:] + self.edges[:-1]) / 2
        self.d_edges = np.diff(self.edges)
        self.d_nodes = np.diff(self.nodes)
        self.npts = self.nodes.size
        self.coord_sys = coord_sys

        # Add tab locations in terms of "left" and "right"
        if tabs:
            self.tabs = {}
            l_z = self.edges[-1]

            def near(x, point, tol=3e-16):
                return abs(x - point) < tol

            if near(tabs["negative"]["z_centre"], 0):
                self.tabs["negative tab"] = "left"
            elif near(tabs["negative"]["z_centre"], l_z):
                self.tabs["negative tab"] = "right"

            if near(tabs["positive"]["z_centre"], 0):
                self.tabs["positive tab"] = "left"
            elif near(tabs["positive"]["z_centre"], l_z):
                self.tabs["positive tab"] = "right"


class Uniform1DSubMesh(SubMesh1D):
    """
    A class to generate a uniform submesh on a 1D domain

    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of the spatial variables
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs
    """

    def __init__(self, lims, npts, tabs=None):

        # currently accept lims and npts as dicts. This may get changed at a future
        # date depending on the form of mesh we desire for 2D/3D

        # check that only one variable passed in
        if len(lims) != 1:
            raise pybamm.GeometryError("lims should only contain a single variable")

        spatial_var = list(lims.keys())[0]
        spatial_lims = lims[spatial_var]
        npts = npts[spatial_var.id]

        edges = np.linspace(spatial_lims["min"], spatial_lims["max"], npts + 1)

        coord_sys = spatial_var.coord_sys

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)
