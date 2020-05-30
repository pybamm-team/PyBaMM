#
# One-dimensional submeshes
#
import pybamm
from .meshes import SubMesh

import numpy as np


class SubMesh1D(SubMesh):
    """
    1D submesh class.
    Contains the position of the nodes, the number of mesh points, and
    (optionally) information about the tab locations.

    Parameters
    ----------
    edges : array_like
        An array containing the points corresponding to the edges of the submesh
    coord_sys : string
        The coordinate system of the submesh
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs

    **Extends:"": :class:`pybamm.SubMesh`
    """

    def __init__(self, edges, coord_sys, tabs=None):
        self.edges = edges
        self.nodes = (self.edges[1:] + self.edges[:-1]) / 2
        self.d_edges = np.diff(self.edges)
        self.d_nodes = np.diff(self.nodes)
        self.npts = self.nodes.size
        self.coord_sys = coord_sys
        self.internal_boundaries = []

        # Add tab locations in terms of "left" and "right"
        if tabs:
            self.tabs = {}
            l_z = self.edges[-1]

            def near(x, point, tol=3e-16):
                return abs(x - point) < tol

            for tab in ["negative", "positive"]:
                tab_location = tabs[tab]["z_centre"]
                if near(tab_location, 0):
                    self.tabs[tab + " tab"] = "left"
                elif near(tab_location, l_z):
                    self.tabs[tab + " tab"] = "right"
                else:
                    raise pybamm.GeometryError(
                        """{} tab located at {}, but must be at either 0 or {}
                        (in dimensionless coordinates).""".format(
                            tab, tab_location, l_z
                        )
                    )

    def read_lims(self, lims):
        # Separate limits and tabs
        # Read and remove tabs. If "tabs" is not a key in "lims", then tabs is set to
        # "None" and nothing is removed from lims
        tabs = lims.pop("tabs", None)

        # check that only one variable passed in
        if len(lims) != 1:
            raise pybamm.GeometryError("lims should only contain a single variable")

        ((spatial_var, spatial_lims),) = lims.items()
        return spatial_var, spatial_lims, tabs


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

    **Extends:"": :class:`pybamm.SubMesh1D`
    """

    def __init__(self, lims, npts):

        spatial_var, spatial_lims, tabs = self.read_lims(lims)
        npts = npts[spatial_var.id]

        edges = np.linspace(spatial_lims["min"], spatial_lims["max"], npts + 1)

        coord_sys = spatial_var.coord_sys

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)


class Exponential1DSubMesh(SubMesh1D):
    """
    A class to generate a submesh on a 1D domain in which the points are clustered
    close to one or both of boundaries using an exponential formula on the interval
    [a,b].

    If side is "left", the gridpoints are given by

    .. math::
        x_{k} = (b-a) + \\frac{\\exp{\\alpha k / N} - 1}{\\exp{\\alpha} - 1}} + a,

    for k = 1, ..., N, where N is the number of nodes.

    Is side is "right", the gridpoints are given by

    .. math::
        x_{k} = (b-a) + \\frac{\\exp{-\\alpha k / N} - 1}{\\exp{-\\alpha} - 1}} + a,

    for k = 1, ..., N.

    If side is "symmetric", the first half of the interval is meshed using the
    gridpoints

    .. math::
        x_{k} = (b/2-a) + \\frac{\\exp{\\alpha k / N} - 1}{\\exp{\\alpha} - 1}} + a,

    for k = 1, ..., N. The grid spacing is then reflected to contruct the grid
    on the full interval [a,b].

    In the above, alpha is a stretching factor. As the number of gridpoints tends
    to infinity, the ratio of the largest and smallest grid cells tends to exp(alpha).

    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of the spatial variables
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable. Note: the number of nodes (located at the cell centres)
        is npts, and the number of edges is npts+1.
    side : str, optional
        Whether the points are clustered near to the left or right boundary,
        or both boundaries. Can be "left", "right" or "symmetric". Default is
        "symmetric"
    stretch : float, optional
        The factor (alpha) which appears in the exponential. If side is "symmetric"
        then the default stretch is 1.15. If side is "left" or "right" then the
        default stretch is 2.3.

    **Extends:"": :class:`pybamm.SubMesh1D`
    """

    def __init__(self, lims, npts, side="symmetric", stretch=None):

        spatial_var, spatial_lims, tabs = self.read_lims(lims)
        a = spatial_lims["min"]
        b = spatial_lims["max"]
        npts = npts[spatial_var.id]
        coord_sys = spatial_var.coord_sys

        # Set stretch if not provided
        if not stretch:
            if side == "symmetric":
                stretch = 1.15
            elif side in ["left", "right"]:
                stretch = 2.3

        # Create edges accoriding to "side"
        if side == "left":
            ii = np.array(range(0, npts + 1))
            edges = (b - a) * (np.exp(stretch * ii / npts) - 1) / (
                np.exp(stretch) - 1
            ) + a

        elif side == "right":
            ii = np.array(range(0, npts + 1))
            edges = (b - a) * (np.exp(-stretch * ii / npts) - 1) / (
                np.exp(-stretch) - 1
            ) + a

        elif side == "symmetric":
            # Mesh half-interval [a, b/2]
            if npts % 2 == 0:
                ii = np.array(range(0, int((npts) / 2)))
            else:
                ii = np.array(range(0, int((npts + 1) / 2)))
            x_exp_left = (b / 2 - a) * (np.exp(stretch * ii / npts) - 1) / (
                np.exp(stretch) - 1
            ) + a

            # Refelct mesh
            x_exp_right = b * np.ones_like(x_exp_left) - (x_exp_left[::-1] - a)

            # Combine left and right halves of the mesh, adding a node at the
            # centre if npts is even (odd number of edges)
            if npts % 2 == 0:
                edges = np.concatenate((x_exp_left, [(a + b) / 2], x_exp_right))
            else:
                edges = np.concatenate((x_exp_left, x_exp_right))

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)


class Chebyshev1DSubMesh(SubMesh1D):
    """
    A class to generate a submesh on a 1D domain using Chebyshev nodes on the
    interval (a, b), given by

    .. math::
        x_{k} = \\frac{1}{2}(a+b) + \\frac{1}{2}(b-a) \\cos(\\frac{2k-1}{2N}\\pi),

    for k = 1, ..., N, where N is the number of nodes. Note: this mesh then
    appends the boundary edges, so that the mesh edges are given by

    .. math::
        a < x_{1} < ... < x_{N} < b.

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

    **Extends:"": :class:`pybamm.SubMesh1D`
    """

    def __init__(self, lims, npts, tabs=None):

        spatial_var, spatial_lims, tabs = self.read_lims(lims)
        npts = npts[spatial_var.id]

        # Create N Chebyshev nodes in the interval (a,b)
        N = npts - 1
        ii = np.array(range(1, N + 1))
        a = spatial_lims["min"]
        b = spatial_lims["max"]
        x_cheb = (a + b) / 2 + (b - a) / 2 * np.cos((2 * ii - 1) * np.pi / 2 / N)

        # Append the boundary nodes. Note: we need to flip the order the Chebyshev
        # nodes as they are created in descending order.
        edges = np.concatenate(([a], np.flip(x_cheb), [b]))
        coord_sys = spatial_var.coord_sys

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)


class UserSupplied1DSubMesh(SubMesh1D):
    """
    A class to generate a submesh on a 1D domain from a user supplied array of
    edges.

    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of the spatial variables
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable. Note: the number of nodes (located at the cell centres)
        is npts, and the number of edges is npts+1.
    edges : array_like
        The array of points which correspond to the edges of the mesh.

    **Extends:"": :class:`pybamm.SubMesh1D`
    """

    def __init__(self, lims, npts, edges=None):

        # raise error if no edges passed
        if edges is None:
            raise pybamm.GeometryError("User mesh requires parameter 'edges'")

        spatial_var, spatial_lims, tabs = self.read_lims(lims)
        npts = npts[spatial_var.id]

        # check that npts + 1 equals number of user-supplied edges
        if (npts + 1) != len(edges):
            raise pybamm.GeometryError(
                """User-suppled edges has should have length (npts + 1) but has length {}.
                 Number of points (npts) for domain {} is {}.""".format(
                    len(edges), spatial_var.domain, npts
                )
            )

        # check end points of edges agree with spatial_lims
        if edges[0] != spatial_lims["min"]:
            raise pybamm.GeometryError(
                """First entry of edges is {}, but should be equal to {}
                 for domain {}.""".format(
                    edges[0], spatial_lims["min"], spatial_var.domain
                )
            )
        if edges[-1] != spatial_lims["max"]:
            raise pybamm.GeometryError(
                """Last entry of edges is {}, but should be equal to {}
                for domain {}.""".format(
                    edges[-1], spatial_lims["max"], spatial_var.domain
                )
            )

        coord_sys = spatial_var.coord_sys

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)
