#
# Spatial operators (grad, div, etc)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np

KNOWN_DOMAINS = ["xc", "xcn", "xcs", "xcp"]


class Operators(object):
    """Contains functions that calculate the spatial derivatives.

    Parameters
    ----------
    spatial_discretisation : string
        The spatial discretisation scheme (see pybamm.solver).
    domain : string
        The domain in which the operators should be calculated.
    mesh : :class:`pybamm.mesh.Mesh' instance
        The mesh used for the spatial discretisation.

    """

    def __init__(self, spatial_discretisation, domain, mesh):
        self.spatial_discretisation = spatial_discretisation
        if domain not in KNOWN_DOMAINS:
            raise NotImplementedError(
                """Domain '{}' is not implemented.
                   Valid choices: one of '{}'.""".format(
                    domain, KNOWN_DOMAINS
                )
            )
        self.domain = domain

        self.mesh = mesh

    def grad(self, y):
        """Calculates the 1D gradient using Finite Volumes.

        Parameters
        ----------
        y : array_like, shape (n,)
            The variable whose gradient is to be calculated.

        Returns
        -------
        array_like, shape (n-1,)
            The gradient, grad(y).

        """
        if self.spatial_discretisation == "Finite Volumes":
            if self.domain == "xc":
                xc, dxc = self.mesh.xc, self.mesh.dxc
            elif self.domain == "xcn":
                xc, dxc = self.mesh.xcn, self.mesh.dxn
            elif self.domain == "xcp":
                xc, dxc = self.mesh.xcp, self.mesh.dxp
            # Run some basic checks on inputs
            assert (
                y.shape == xc.shape
            ), """xc and y should have the same shape,
                but xc.shape = {} and yc.shape = {}""".format(
                xc.shape, y.shape
            )

            # Calculate internal flux
            return np.diff(y) / dxc

    def div(self, N):
        """Calculates the 1D divergence using Finite Volumes.

        Parameters
        ----------
        N : array_like, shape (n,)
            The flux whose divergence is to be calculated.

        Returns
        -------
        array_like, shape (n-1,)
            The divergence, div(N).

        """
        if self.spatial_discretisation == "Finite Volumes":
            if self.domain == "xc":
                x, dx = self.mesh.x, self.mesh.dx
            elif self.domain == "xcn":
                x, dx = self.mesh.xn, self.mesh.dxn
            elif self.domain == "xcp":
                x, dx = self.mesh.xp, self.mesh.dxp

            # Run basic checks on inputs
            assert (
                N.shape == x.shape
            ), """x and N should have the same shape,
                but x.shape = {} and N.shape = {}""".format(
                x.shape, N.shape
            )

            return np.diff(N) / dx
