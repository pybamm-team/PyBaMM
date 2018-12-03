#
# Spatial operators (grad, div, etc)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class Operators(object):
    """Contains functions that calculate the spatial derivatives.

    Parameters
    ----------
    spatial_discretisation : string
        The spatial discretisation scheme (see pybamm.solver).
    mesh : :class:`pybamm.mesh.Mesh` instance
        The mesh used for the spatial discretisation.

    """

    def __init__(self, spatial_discretisation, mesh):
        self.spatial_discretisation = spatial_discretisation
        self.mesh = mesh

    @property
    def xn(self):
        if self.spatial_discretisation == "Finite Volumes":
            return CartesianFiniteVolumes(self.mesh.xn)

    @property
    def xp(self):
        if self.spatial_discretisation == "Finite Volumes":
            return CartesianFiniteVolumes(self.mesh.xp)

    @property
    def x(self):
        if self.spatial_discretisation == "Finite Volumes":
            return CartesianFiniteVolumes(self.mesh.x)

    # @property
    # def rn(self):
    #     if self.spatial_discretisation == "Finite Volumes":
    #         return SphericalFiniteVolumes(self.mesh.rn)
    #
    # @property
    # def rp(self):
    #     if self.spatial_discretisation == "Finite Volumes":
    #         return SphericalFiniteVolumes(self.mesh.rp)


class CartesianFiniteVolumes(object):
    def __init__(self, submesh):
        self.submesh = submesh

    def grad(self, y):
        """Calculate the 1D gradient using Finite Volumes.

        Parameters
        ----------
        y : array_like, shape (n,)
            The variable whose gradient is to be calculated.

        Returns
        -------
        array_like, shape (n-1,)
            The gradient, grad(y).

        """
        # Run basic checks on input
        assert (
            y.shape == self.submesh.centres.shape
        ), """xc and y should have the same shape,
            but xc.shape = {} and yc.shape = {}""".format(
            self.submesh.centres.shape, y.shape
        )

        # Calculate internal flux
        return np.diff(y) / self.submesh.d_centres

    def div(self, N):
        """Calculate the 1D divergence using Finite Volumes.

        Parameters
        ----------
        N : array_like, shape (n,)
            The flux whose divergence is to be calculated.

        Returns
        -------
        array_like, shape (n-1,)
            The divergence, div(N).

        """
        # Run basic checks on inputs
        assert (
            N.shape == self.submesh.edges.shape
        ), """x and N should have the same shape,
            but x.shape = {} and N.shape = {}""".format(
            self.submesh.edges.shape, N.shape
        )

        return np.diff(N) / self.submesh.d_edges
