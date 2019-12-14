#
# Spatial method for zero dimensional meshes
#
import pybamm
import numpy as np


class ZeroDimensionalMethod(pybamm.SpatialMethod):
    """
    A discretisation class for the zero dimensional mesh

    Parameters
    ----------
    mesh : :class: `pybamm.Mesh`
        Contains all the submeshes for discretisation

    **Extends** : :class:`pybamm.SpatialMethod`
    """

    def __init__(self, options=None):
        super().__init__(options)

    def build(self, mesh):
        self._mesh = mesh

    def boundary_value_or_flux(self, symbol, discretised_child, bcs=None):
        """
        In 0D, the boundary value is the identity operator.
        See :meth:`SpatialMethod.boundary_value_or_flux`
        """
        return discretised_child

    def mass_matrix(self, symbol, boundary_conditions):
        """
        Calculates the mass matrix for a spatial method. Since the spatial method is
        zero dimensional, this is simply the number 1.
        """
        return pybamm.Matrix(np.ones((1, 1)))
