#
# Chebyshev one-dimensional submesh
#
import pybamm

import numpy as np


class Chebyshev1DSubMesh(pybamm.SubMesh1D):
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
