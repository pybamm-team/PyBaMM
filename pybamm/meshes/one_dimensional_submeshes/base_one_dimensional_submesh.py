#
# Base class for one-dimensional submeshes
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

            for tab in ["negative", "positive"]:
                if near(tabs[tab]["z_centre"], 0):
                    self.tabs[tab + " tab"] = "left"
                elif near(tabs[tab]["z_centre"], l_z):
                    self.tabs[tab + " tab"] = "right"
