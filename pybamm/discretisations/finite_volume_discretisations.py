#
# Finite Volume discretisation class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
from scipy.sparse import spdiags


class FiniteVolumeDiscretisation(pybamm.BaseDiscretisation):
    """Discretisation using Finite Volumes.

    Parameters
    ----------
    mesh : :class:`pybamm.BaseMesh` (or subclass)
        The underlying mesh for discretisation

    **Extends:** :class:`pybamm.BaseDiscretisation`
    """

    def __init__(self, mesh):
        super().__init__(mesh)

    def gradient(self, symbol, domain, y_slices, boundary_conditions):
        """Matrix-vector multiplication to implement the gradient operator.
        See :meth:`pybamm.BaseDiscretisation.gradient`
        """
        # Check that boundary condition keys are hashes (ids)
        for key in boundary_conditions.keys():
            assert isinstance(key, int), TypeError(
                "boundary condition keys should be hashes, not {}".format(type(key))
            )
        # Discretise symbol
        discretised_symbol = self.process_symbol(
            symbol, domain, y_slices, boundary_conditions
        )
        # Add boundary conditions if defined
        if symbol.id in boundary_conditions:
            lbc = boundary_conditions[symbol.id]["left"]
            rbc = boundary_conditions[symbol.id]["right"]
            discretised_symbol = self.concatenate(lbc, discretised_symbol, rbc)
        gradient_matrix = self.gradient_matrix(domain)
        return gradient_matrix * discretised_symbol

    def gradient_matrix(self, domain):
        """
        Gradient matrix for finite volumes in the appropriate domain.
        Equivalent to grad(y) = (y[1:] - y[:-1])/dx

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the gradient matrix
        """
        assert len(domain) == 1
        # TODO: implement for when there are several domains

        # implementation for a single domain
        submesh = self.mesh[domain[0]]
        n = submesh.npts
        e = 1 / submesh.d_nodes
        data = np.vstack(
            [np.concatenate([-e, np.array([0])]), np.concatenate([np.array([0]), e])]
        )
        diags = np.array([0, 1])
        matrix = spdiags(data, diags, n - 1, n)
        return pybamm.Matrix(matrix)

    def divergence(self, symbol, domain, y_slices, boundary_conditions):
        """Matrix-vector multiplication to implement the divergence operator.
        See :meth:`pybamm.BaseDiscretisation.gradient`
        """
        # Check that boundary condition keys are hashes (ids)
        for key in boundary_conditions.keys():
            assert isinstance(key, int), TypeError(
                "boundary condition keys should be hashes, not {}".format(type(key))
            )
        # Discretise symbol
        discretised_symbol = self.process_symbol(
            symbol, domain, y_slices, boundary_conditions
        )
        # Add boundary conditions if defined
        if symbol.id in boundary_conditions:
            lbc = boundary_conditions[symbol.id]["left"]
            rbc = boundary_conditions[symbol.id]["right"]
            discretised_symbol = self.concatenate(lbc, discretised_symbol, rbc)
        divergence_matrix = self.divergence_matrix(domain)
        return divergence_matrix * discretised_symbol

    def divergence_matrix(self, domain):
        """
        Divergence matrix for finite volumes in the appropriate domain.
        Equivalent to div(N) = (N[1:] - N[:-1])/dx

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the divergence matrix
        """
        assert len(domain) == 1
        # TODO: implement for when there are several domains

        # implementation for a single domain
        submesh = self.mesh[domain[0]]
        n = submesh.npts + 1
        e = 1 / submesh.d_edges
        data = np.vstack(
            [np.concatenate([-e, np.array([0])]), np.concatenate([np.array([0]), e])]
        )
        diags = np.array([0, 1])
        matrix = spdiags(data, diags, n - 1, n)
        return pybamm.Matrix(matrix)
