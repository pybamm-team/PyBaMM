#
# Mesh class for space and time discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class Mesh(object):
    """A 1D mesh for Finite Volumes.

    Parameters
    ----------
    param : :class:`pybamm.parameters.Parameters` instance
        The parameters defining the subdomain sizes.
    target_npts : int
        The target number of points in each domain. The mesh will be created
        in such a way that the cell sizes are as similar as possible between
        domains.
    tsteps : int
        The number of time steps to take
    tend : float
        The finishing time for the simulation

    """

    def __init__(self, param, target_npts=10, tsteps=100, tend=1):

        # Time
        self.time = np.linspace(0, tend, tsteps)

        # Space (macro)
        ln, ls, lp = param.geometric.ln, param.geometric.ls, param.geometric.lp
        # We aim to create the grid as uniformly as possible
        targetmeshsize = min(ln, ls, lp) / target_npts

        # Negative electrode
        self.nn = round(ln / targetmeshsize) + 1
        self.set_submesh("xn", np.linspace(0.0, ln, self.nn))

        # Separator
        self.ns = round(ls / targetmeshsize) + 1
        self.set_submesh("xs", np.linspace(ln, ln + ls, self.ns))
        # Positive electrode
        self.np = round(lp / targetmeshsize) + 1
        self.set_submesh("xp", np.linspace(ln + ls, 1.0, self.np))

        # Totals
        self.n = self.nn + (self.ns - 2) + self.np
        self.set_submesh(
            "x", np.concatenate([self.xn.edges, self.xs.edges[1:-1], self.xp.edges])
        )

        # Space (micro)
        # TODO: write this

    @property
    def xn(self):
        return self._xn

    @property
    def xs(self):
        return self._xs

    @property
    def xp(self):
        return self._xp

    @property
    def x(self):
        return self._x

    def set_submesh(self, submesh, edges):
        self.__dict__["_" + submesh] = _SubMesh(edges)


class _SubMesh:
    def __init__(self, edges):
        self.edges = edges
        self.centres = (self.edges[1:] + self.edges[:-1]) / 2
        self.d_edges = np.diff(self.edges)
        self.d_centres = np.diff(self.centres)
        self.npts = self.centres.size
