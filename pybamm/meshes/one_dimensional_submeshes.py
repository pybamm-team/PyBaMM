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

            for tab in ["negative", "positive"]:
                if near(tabs[tab]["z_centre"], 0):
                    self.tabs[tab + " tab"] = "left"
                elif near(tabs[tab]["z_centre"], l_z):
                    self.tabs[tab + " tab"] = "right"


class Uniform1DSubMesh(SubMesh1D):
    """
    A class to generate a uniform submesh on a 1D domain

    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of the spatial variables
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable. Note: the number of nodes (located at the cell centres)
        is npts, and the number of edges is npts+1.
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs
    """

    def __init__(self, lims, npts, tabs=None):

        # check that only one variable passed in
        if len(lims) != 1:
            raise pybamm.GeometryError("lims should only contain a single variable")

        spatial_var = list(lims.keys())[0]
        spatial_lims = lims[spatial_var]
        npts = npts[spatial_var.id]

        edges = np.linspace(spatial_lims["min"], spatial_lims["max"], npts + 1)

        coord_sys = spatial_var.coord_sys

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)


class GetUserSupplied1DSubMesh:
    """
    A class to generate a submesh on a 1D domain using a user supplied vector of
    nodes.

    Parameters
    ----------
    edges : array_like
        The array of points which correspond to the edges of the mesh.


    """

    def __init__(self, nodes):
        self.nodes = nodes

    def __call__(self, lims, npts, tabs=None):
        return UserSupplied1DSubMesh(lims, npts, tabs, self.nodes)


class UserSupplied1DSubMesh(SubMesh1D):
    """
    A class to generate a submesh on a 1D domain from a user supplied array of
    nodes.

    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of the spatial variables
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable. Note: the number of nodes (located at the cell centres)
        is npts, and the number of edges is npts+1.
    tabs : dict
        A dictionary that contains information about the size and location of
        the tabs
    edges : array_like
        The array of points which correspond to the edges of the mesh.
    """

    def __init__(self, lims, npts, tabs, edges):

        # check that only one variable passed in
        if len(lims) != 1:
            raise pybamm.GeometryError("lims should only contain a single variable")

        spatial_var = list(lims.keys())[0]
        spatial_lims = lims[spatial_var]
        npts = npts[spatial_var.id]
        import ipdb

        ipdb.set_trace()
        # check that npts + 1 equals number of user-supplied edges
        if (npts + 1) != len(edges):
            raise pybamm.GeometryError(
                """User-suppled edges has should have length (npts + 1) but has length {}.
                 Number of points (npts) for domain {} is {}.""".format(
                    len(edges), spatial_var.domain, npts
                )
            )

        # check end points of edges agrees with spatial_lims
        if edges[0] != spatial_lims["min"]:
            raise pybamm.GeometryError(
                "First entry of edges is , but should be equal to {} for domain {}.".format(
                    edges[0], spatial_lims["min"], spatial_var.domain
                )
            )
        if edges[-1] != spatial_lims["max"]:
            raise pybamm.GeometryError(
                "Last entry of edges is , but should be equal to {} for domain {}.".format(
                    edges[-1], spatial_lims["max"], spatial_var.domain
                )
            )

        coord_sys = spatial_var.coord_sys

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)
