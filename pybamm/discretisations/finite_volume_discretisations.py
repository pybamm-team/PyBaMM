#
# Finite Volume discretisation class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
from scipy.sparse import spdiags


class FiniteVolumeDiscretisation(pybamm.MatrixVectorDiscretisation):
    """Discretisation using Finite Volumes.
    Inherits from :class:`pybamm.MatrixVectorDiscretisation`, so we only need to
    implement the gradient and divergence matrices

    Parameters
    ----------
     mesh : :class:`BaseMesh` (or subclass)
            The underlying mesh for discretisation

    **Extends:** :class:`pybamm.MatrixVectorDiscretisation`
    """

    def __init__(self, mesh):
        super().__init__(mesh)

    def gradient_matrix(self, domain):
        """
        Gradient matrix for finite volumes in the appropriate domain.
        Equivalent to grad(y) = (y[1:] - y[:-1])/dx
        See :meth:`pybamm.MatrixVectorDiscretisation.gradient_matrix()`
        """
        assert len(domain) == 1
        # TODO: implement for when there are several domains

        # implementation for a single domain
        submesh = getattr(self.mesh, domain[0])
        n = submesh.npts
        e = 1 / submesh.d_nodes
        data = np.vstack(
            [np.concatenate([-e, np.array([0])]), np.concatenate([np.array([0]), e])]
        )
        diags = np.array([0, 1])
        matrix = spdiags(data, diags, n - 1, n)
        return pybamm.Matrix(matrix)

    def divergence_matrix(self, domain):
        """
        Divergence matrix for finite volumes in the appropriate domain.
        Equivalent to div(N) = (N[1:] - N[:-1])/dx
        See :meth:`pybamm.MatrixVectorDiscretisation.divergence_matrix()`
        """
        assert len(domain) == 1
        # TODO: implement for when there are several domains

        # implementation for a single domain
        submesh = getattr(self.mesh, domain[0])
        n = submesh.npts + 1
        e = 1 / submesh.d_edges
        data = np.vstack(
            [np.concatenate([-e, np.array([0])]), np.concatenate([np.array([0]), e])]
        )
        diags = np.array([0, 1])
        matrix = spdiags(data, diags, n - 1, n)
        return pybamm.Matrix(matrix)
