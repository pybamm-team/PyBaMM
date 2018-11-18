#
# Equations for the electrolyte
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class Electrolyte(object):
    """Equations for the electrolyte."""

    def set_simulation(self, param, operators, mesh):
        """
        Assign simulation-specific objects as attributes.

        Parameters
        ----------
        param : :class:`pybamm.Parameters` instance
            The parameters of the simulation
        operators : :class:`pybamm.Operators` instance
            The spatial operators.
        mesh : :class:`pybamm.Mesh` instance
            The spatial and temporal discretisation.
        """
        self.param = param
        self.operators = operators
        self.mesh = mesh

    def initial_conditions(self):
        """Calculates initial conditions for variables in the electrolyte.

        Returns
        -------
        dict
            The initial conditions
        """
        c0 = self.param.c0 * np.ones_like(self.mesh.xc)

        return {"c": c0}

    def cation_conservation(self, c, j, flux_bcs):
        """Conservation of cations.

        Parameters
        ----------
        c : array_like, shape (n,)
            The electrolyte concentration.
        j : array_like, shape (n,)
            The interfacial current density.
        flux_bcs : 2-tuple of array_like, shape (1,)
            Flux at the boundaries (Neumann BC).

        Returns
        -------
        dcdt : array_like, shape (n,)
            The time derivative of c.

        """
        # Calculate internal flux
        N_internal = -self.operators["xc"].grad(c)

        # Add boundary conditions (Neumann)
        flux_bc_left, flux_bc_right = flux_bcs
        N = np.concatenate([flux_bc_left, N_internal, flux_bc_right])

        # Calculate time derivative
        dcdt = -self.operators["xc"].div(N) + self.param.s * j

        return dcdt

    def bcs_cation_flux(self):
        """Flux boundary conditions for the cation conservation equation.

        Returns
        -------
        2-tuple of array_like, shape(1,)
            The boundary conditions.

        """
        flux_bc_left = np.array([0])
        flux_bc_right = np.array([0])
        return (flux_bc_left, flux_bc_right)

    # def current_conservation(self, c, e, j, bcs=None):
    #     """The 1D diffusion equation.
    #
    #     Parameters
    #     ----------
    #     param : pybamm.parameters.Parameter() instance
    #         The parameters of the simulation
    #     variables : 2-tuple (c, e) of array_like, shape (n,)
    #         The concentration, and potential difference.
    #     operators : pybamm.operators.Operators() instance
    #         The spatial operators.
    #     current_bcs : 2-tuple of array_like, shape (1,)
    #         Flux at the boundaries (Neumann BC).
    #     j : array_like, shape (n,)
    #         Interfacial current density.
    #
    #     Returns
    #     -------
    #     dedt : array_like, shape (n,)
    #         The time derivative of the potential.
    #
    #     """
    #     # Calculate current density
    #     i = self.macinnes(variables, operators, current_bcs)
    #
    #     # Calculate time derivative
    #     dedt = 1 / param.gamma_dl * (operators.div(i) - j)
    #
    #     return dedt
    #
    # def macinnes(self, c, e, bcs=None):
    #     """The 1D current.
    #
    #     Parameters
    #     ----------
    #     param : pybamm.parameters.Parameter() instance
    #         The parameters of the simulation
    #     variables : 2-tuple (c, e) of array_like, shape (n,)
    #         The concentration, and potential difference.
    #     operators : pybamm.operators.Operators() instance
    #         The spatial operators.
    #     current_bcs : 2-tuple of array_like, shape (1,)
    #         Flux at the boundaries (Neumann BC).
    #
    #     Returns
    #     -------
    #     i : array_like, shape (n+1,)
    #         The current density.
    #     """
    #     c, e = variables
    #
    #     kappa_over_c = 1
    #     kappa = 1
    #
    #     # Calculate inner current
    #     i_inner = kappa_over_c * operators.grad(c) + kappa * operators.grad(e
    #
    #     # Add boundary conditions
    #     lbc, rbc = current_bcs
    #     i = np.concatenate([lbc, i_inner, rbc])
    #
    #     return i
