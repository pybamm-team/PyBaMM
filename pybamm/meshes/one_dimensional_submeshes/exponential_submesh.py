#
# Exponential formula for one-dimensional submesh
#
import pybamm

import numpy as np


class GetExponential1DSubMesh:
    """
    A class to generate a submesh on a 1D domain in which the points are clustered
    close to one or both boundaries using an exponential formula on the interval [a,b].

    Parameters
    ----------
    side : str, optional
        Whether the points are clustered near to the left or right boundary,
        or both boundaries. Can be "left", "right" or "symmetric". Defualt is
        "symmetric".
    stretch : float, optional
        The factor which appears in the exponential, defualt is 1.15 is side is
        "symmetric" and 2.3 is side is "left" or "right".

    """

    def __init__(self, side="symmetric", stretch=None):
        self.side = side
        if stretch:
            self.stretch = stretch
        elif side == "symmetric":
            self.stretch = 1.15
        elif side in ["left", "right"]:
            self.stretch = 2.3

    def __call__(self, lims, npts, tabs=None):
        return Exponential1DSubMesh(lims, npts, tabs, self.side, self.stretch)


class Exponential1DSubMesh(pybamm.SubMesh1D):
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
    side : str
        Whether the points are clustered near to the left or right boundary,
        or both boundaries. Can be "left", "right" or "symmetric"
    tabs : dict
        A dictionary that contains information about the size and location of
        the tabs
    stretch : float
        The factor which appears in the exponential.
    """

    def __init__(self, lims, npts, tabs, side, stretch):

        # check that only one variable passed in
        if len(lims) != 1:
            raise pybamm.GeometryError("lims should only contain a single variable")

        spatial_var = list(lims.keys())[0]
        spatial_lims = lims[spatial_var]
        a = spatial_lims["min"]
        b = spatial_lims["max"]
        npts = npts[spatial_var.id]
        coord_sys = spatial_var.coord_sys

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
