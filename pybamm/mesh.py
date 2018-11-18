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
    param : pybamm.parameters.Parameters() instance
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

    def __init__(self, param, target_npts, tsteps=100, tend=1):
        # Space
        # We aim to create the grid as uniformly as possible
        targetmeshsize = min(param.ln, param.ls, param.lp) / target_npts

        # Negative electrode
        self.nn = round(param.ln / targetmeshsize) + 1
        self.dxn = param.ln / (self.nn - 1)
        # Separator
        self.ns = round(param.ls / targetmeshsize) - 1
        self.dxs = param.ls / (self.ns + 1)
        # Positive electrode
        self.np = round(param.lp / targetmeshsize) + 1
        self.dxp = param.lp / (self.np - 1)
        # Totals
        self.n = self.nn + self.ns + self.np

        # Grid: edges
        self.xn = np.linspace(0.0, param.ln, self.nn)
        self.xs = np.linspace(
            param.ln + self.dxs, param.ln + param.ls - self.dxs, self.ns
        )
        self.xp = np.linspace(param.ln + param.ls, 1.0, self.np)
        self.x = np.concatenate([self.xn, self.xs, self.xp])
        self.dx = np.diff(self.x)

        # Grid: centres
        self.xcn = (self.xn[1:] + self.xn[:-1]) / 2
        self.xcs = np.linspace(
            param.ln + self.dxs / 2,
            param.ln + param.ls - self.dxs / 2,
            self.ns + 1,
        )
        self.xcp = (self.xp[1:] + self.xp[:-1]) / 2
        self.xc = (self.x[1:] + self.x[:-1]) / 2
        self.dxc = np.diff(self.xc)

        # Time
        self.time = np.linspace(0, tend, tsteps)
