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
        # Add Dirichlet boundary conditions, if defined
        if symbol.id in boundary_conditions:
            lbc = boundary_conditions[symbol.id]["left"]
            rbc = boundary_conditions[symbol.id]["right"]
            discretised_symbol = self.add_ghost_nodes(discretised_symbol, lbc, rbc)
            domain_ = (
                [domain[0] + "_left_ghost_cell"]
                + domain
                + [domain[-1] + "_right_ghost_cell"]
            )
        gradient_matrix = self.gradient_matrix(domain)
        return gradient_matrix * discretised_symbol

    def add_ghost_nodes(self, discretised_symbol, lbc, rbc):
        """
        Add Dirichlet boundary conditions via ghost nodes.

        For a boundary condition
          y = a at x=0 ("left" boundary),
        we concatenate a ghost node to the start of the vector y with value
          2*a - y1
        where y1 is the value of the first node.
        Similarly for the right-hand boundary condition.

        Currently, Dirichlet boundary conditions can only be applied on state
        variables (e.g. concentration, temperature), and not on expressions.
        To access the value of the first node (y1), we create a "first_node" object
        which is a StateVector whose y_slice is the start of the y_slice of
        discretised_symbol.
        Similarly, the last node is a StateVector whose y_slice is the end of the
        y_slice of discretised_symbol

        Parameters
        ----------
        discretised_symbol : :class:`pybamm.StateVector` (size n)
            The discretised variable (a state vector) to which to add ghost nodes
        lbc : :class:`pybamm.Scalar`
            Dirichlet bouncary condition on the left-hand side
        rbc : :class:`pybamm.Scalar`
            Dirichlet bouncary condition on the right-hand side

        Returns
        -------
        :class:`pybamm.Concatenation` (size n+2)
            Concatenation of the variable (a state vector) and ghost nodes

        """
        assert isinstance(discretised_symbol, pybamm.StateVector), NotImplementedError(
            """discretised_symbol must be a StateVector, not {}""".format(
                type(discretised_symbol)
            )
        )
        # left ghost cell
        y_slice_start = discretised_symbol.y_slice.start
        first_node = pybamm.StateVector(slice(y_slice_start, y_slice_start + 1))
        left_ghost_cell = 2 * lbc - first_node
        # right ghost cell
        y_slice_stop = discretised_symbol.y_slice.stop
        last_node = pybamm.StateVector(slice(y_slice_stop - 1, y_slice_stop))
        right_ghost_cell = 2 * rbc - last_node
        # concatenate
        return self.concatenate(left_ghost_cell, discretised_symbol, right_ghost_cell)

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
        # Add Neumann boundary conditions if defined
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
