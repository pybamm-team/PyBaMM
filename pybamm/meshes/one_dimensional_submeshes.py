#
# One-dimensional submeshes
#
import pybamm

import numpy as np


class SubMesh1D:
    """
    1D submesh class.
    Contains the position of the nodes and the number of mesh points.
    """

    def __init__(self, edges, coord_sys):
        self.edges = edges
        self.nodes = (self.edges[1:] + self.edges[:-1]) / 2
        self.d_edges = np.diff(self.edges)
        self.d_nodes = np.diff(self.nodes)
        self.npts = self.nodes.size
        self.coord_sys = coord_sys


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
    """

    def __init__(self, lims, npts):

        # check that only one variable passed in
        if len(lims) != 1:
            raise pybamm.GeometryError("lims should only contain a single variable")

        spatial_var = list(lims.keys())[0]
        spatial_lims = lims[spatial_var]
        npts = npts[spatial_var.id]

        edges = np.linspace(spatial_lims["min"], spatial_lims["max"], npts + 1)

        coord_sys = spatial_var.coord_sys

        super().__init__(edges, coord_sys=coord_sys)


class Chebyshev1DSubMesh(SubMesh1D):
    """
    A class to generate a submesh on a 1D domain using Chebyshev nodes on the
    interval (a, b), given by

   .. math::
    x_{k} = \\frac{1}{2}(a+b) + \\frac{1}{2}(b-a) \\cos(\\frac{2k-1}{2N}\\pi),

    for k = 1, ..., N, where N is the number of nodes. Note: this mesh then
    appends the boundary nodes, so that the mesh edges are given by

    .. math ::
     a < x_{1} < ... < x_{N} < b.

    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of the spatial variables
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable. Note: the number of nodes (located at the cell centres)
        is npts, and the number of edges is npts+1.
    """

    def __init__(self, lims, npts):

        # check that only one variable passed in
        if len(lims) != 1:
            raise pybamm.GeometryError("lims should only contain a single variable")

        spatial_var = list(lims.keys())[0]
        spatial_lims = lims[spatial_var]
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

        super().__init__(edges, coord_sys=coord_sys)


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

    def __call__(self, lims, npts):
        return Exponential1DSubMesh(lims, npts, self.side, self.stretch)


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
    side : str
        Whether the points are clustered near to the left or right boundary,
        or both boundaries. Can be "left", "right" or "symmetric"
    stretch : float
        The factor which appears in the exponential.
    """

    def __init__(self, lims, npts, side, stretch):

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

        super().__init__(edges, coord_sys=coord_sys)
