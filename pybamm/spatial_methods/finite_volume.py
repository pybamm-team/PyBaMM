#
# Finite Volume discretisation class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
from scipy.sparse import spdiags


class FiniteVolume(pybamm.SpatialMethod):
    """
    A class which implements the steps specific to the finite volume method during 
    discretisation. 

    Parameters
    ----------
    """


    def __init__(self, mesh):
        self.mesh = mesh


     def gradient(self, symbol, y_slices, boundary_conditions):
        """Matrix-vector multiplication to implement the gradient operator.
        See :meth:`pybamm.BaseDiscretisation.gradient`
        """
        # Check that boundary condition keys are hashes (ids)
        for key in boundary_conditions.keys():
            assert isinstance(key, int), TypeError(
                "boundary condition keys should be hashes, not {}".format(type(key))
            )
        # Discretise symbol
        discretised_symbol = self.process_symbol(symbol, y_slices, boundary_conditions)
        domain = symbol.domain
        # Add Dirichlet boundary conditions, if defined
        if symbol.id in boundary_conditions:
            lbc = boundary_conditions[symbol.id]["left"]
            rbc = boundary_conditions[symbol.id]["right"]
            discretised_symbol = self.add_ghost_nodes(discretised_symbol, lbc, rbc)
            domain = (
                [domain[0] + "_left ghost cell"]
                + domain
                + [domain[-1] + "_right ghost cell"]
            )

        # note in 1D spherical grad and normal grad are the same
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

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite volume gradient matrix for the domain
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh = self.mesh.combine_submeshes(*domain)

        # Create matrix using submesh
        n = submesh.npts
        e = 1 / submesh.d_nodes
        data = np.vstack(
            [np.concatenate([-e, np.array([0])]), np.concatenate([np.array([0]), e])]
        )
        diags = np.array([0, 1])
        matrix = spdiags(data, diags, n - 1, n)
        return pybamm.Matrix(matrix)

    def divergence(self, symbol, y_slices, boundary_conditions):
        """Matrix-vector multiplication to implement the divergence operator.
        See :meth:`pybamm.BaseDiscretisation.gradient`
        """
        # Check that boundary condition keys are hashes (ids)
        for key in boundary_conditions.keys():
            assert isinstance(key, int), TypeError(
                "boundary condition keys should be hashes, not {}".format(type(key))
            )
        # Discretise symbol
        discretised_symbol = self.process_symbol(symbol, y_slices, boundary_conditions)
        # Add Neumann boundary conditions if defined
        if symbol.id in boundary_conditions:
            # for the particles there will be a "negative particle" "left" and "right"
            # and also a "positive particle" left and right.
            lbc = boundary_conditions[symbol.id]["left"]
            rbc = boundary_conditions[symbol.id]["right"]
            discretised_symbol = self.concatenate(lbc, discretised_symbol, rbc)

        domain = symbol.domain
        # check for
        if ("negative particle" or "positive particle") in domain:

            # implement spherical operator
            divergence_matrix = self.divergence_matrix(domain)

            submesh = self.mesh.combine_submeshes(*domain)
            r = pybamm.Vector(submesh.nodes)
            r_edges = pybamm.Vector(submesh.edges)

            out = (1 / (r ** 2)) * (
                divergence_matrix * ((r_edges ** 2) * discretised_symbol)
            )

        else:
            divergence_matrix = self.divergence_matrix(domain)
            out = divergence_matrix * discretised_symbol
        return out

    def divergence_matrix(self, domain):
        """
        Divergence matrix for finite volumes in the appropriate domain.
        Equivalent to div(N) = (N[1:] - N[:-1])/dx

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the divergence matrix

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite volume divergence matrix for the domain
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh = self.mesh.combine_submeshes(*domain)

        # Create matrix using submesh
        n = submesh.npts + 1
        e = 1 / submesh.d_edges
        data = np.vstack(
            [np.concatenate([-e, np.array([0])]), np.concatenate([np.array([0]), e])]
        )
        diags = np.array([0, 1])
        matrix = spdiags(data, diags, n - 1, n)
        return pybamm.Matrix(matrix)


