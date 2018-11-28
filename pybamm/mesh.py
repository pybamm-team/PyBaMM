#
# Mesh class for space and time discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class Mesh(object):
    """
    A 1D mesh for Finite Volumes.

    Parameters
    ----------
    param : :class:`pybamm.parameters.Parameters' instance
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
        # Space
        ln, ls, lp = param.geometric["ln"], param.geometric["ls"], param.geometric["lp"]
        # We aim to create the grid as uniformly as possible
        targetmeshsize = min(ln, ls, lp) / target_npts

        # Negative electrode
        self.nn = round(ln / targetmeshsize) + 1
        self.dxn = ln / (self.nn - 1)
        # Separator
        self.ns = round(ls / targetmeshsize) - 1
        self.dxs = ls / (self.ns + 1)
        # Positive electrode
        self.np = round(lp / targetmeshsize) + 1
        self.dxp = lp / (self.np - 1)
        # Totals
        self.n = self.nn + self.ns + self.np

        # Grid: edges
        self.xn = np.linspace(0.0, ln, self.nn)
        self.xs = np.linspace(ln + self.dxs, ln + ls - self.dxs, self.ns)
        self.xp = np.linspace(ln + ls, 1.0, self.np)
        self.x = np.concatenate([self.xn, self.xs, self.xp])
        self.dx = np.diff(self.x)

        # Grid: centres
        self.xcn = (self.xn[1:] + self.xn[:-1]) / 2
        self.xcs = np.linspace(ln + self.dxs / 2, ln + ls - self.dxs / 2, self.ns + 1)
        self.xcp = (self.xp[1:] + self.xp[:-1]) / 2
        self.xc = (self.x[1:] + self.x[:-1]) / 2
        self.dxc = np.diff(self.xc)

        # Time
        self.time = np.linspace(0, tend, tsteps)

    @property
    def whole(self):
        return {"x": self.x, "dx": self.dx, "xc": self.xc, "npts": self.n}

    @property
    def neg(self):
        return {"x": self.xn, "dx": self.dxn, "xc": self.xcn, "npts": self.nn}

    @property
    def sep(self):
        return {"x": self.xs, "dx": self.dxs, "xc": self.xcs, "npts": self.ns}

    @property
    def pos(self):
        return {"x": self.xp, "dx": self.dxp, "xc": self.xcp, "npts": self.np}
