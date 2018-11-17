#
# Equations for the electrolyte
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class Electrolyte:
    def __init__(self):
        pass

    def initial_conditions(param, mesh):
        pass

    def cation_conservation(self, c, j, bcs=None):
        """The 1D diffusion equation.

        Parameters
        ----------
        param : pybamm.parameters.Parameter() instance
            The parameters of the simulation
        c : array_like, shape (n,)
            The electrolyte concentration.
        operators : pybamm.operators.Operators() instance
            The spatial operators.
        flux_bcs : 2-tuple of array_like, shape (1,)
            Flux at the boundaries (Neumann BC).
        j : array_like, shape (n,)
            The interfacial current density.

        Returns
        -------
        dcdt : array_like, shape (n,)
            The time derivative.

        """
        # Calculate internal flux
        N_internal = -operators.grad(c)

        # Add boundary conditions (Neumann)
        flux_bc_left, flux_bc_right = flux_bcs
        N = np.concatenate([flux_bc_left, N_internal, flux_bc_right])

        # Calculate time derivative
        dcdt = -operators.div(N) + param.s * j

        return dcdt

    def current_conservation(self, c, e, j, bcs=None):
        """The 1D diffusion equation.

        Parameters
        ----------
        param : pybamm.parameters.Parameter() instance
            The parameters of the simulation
        variables : 2-tuple (c, e) of array_like, shape (n,)
            The concentration, and potential difference.
        operators : pybamm.operators.Operators() instance
            The spatial operators.
        current_bcs : 2-tuple of array_like, shape (1,)
            Flux at the boundaries (Neumann BC).
        j : array_like, shape (n,)
            Interfacial current density.

        Returns
        -------
        dedt : array_like, shape (n,)
            The time derivative of the potential.

        """
        # Calculate current density
        i = self.macinnes(variables, operators, current_bcs)

        # Calculate time derivative
        dedt = 1 / param.gamma_dl * (operators.div(i) - j)

        return dedt

    def macinnes(self, c, e, bcs=None):
        """The 1D current.

        Parameters
        ----------
        param : pybamm.parameters.Parameter() instance
            The parameters of the simulation
        variables : 2-tuple (c, e) of array_like, shape (n,)
            The concentration, and potential difference.
        operators : pybamm.operators.Operators() instance
            The spatial operators.
        current_bcs : 2-tuple of array_like, shape (1,)
            Flux at the boundaries (Neumann BC).

        Returns
        -------
        i : array_like, shape (n+1,)
            The current density.
        """
        c, e = variables

        kappa_over_c = 1
        kappa = 1

        # Calculate inner current
        i_inner = kappa_over_c * operators.grad(c) + kappa * operators.grad(e)

        # Add boundary conditions
        lbc, rbc = current_bcs
        i = np.concatenate([lbc, i_inner, rbc])

        return i
