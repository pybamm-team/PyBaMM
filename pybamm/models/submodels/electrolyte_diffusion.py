#
# Equation classes for the electrolyte concentration
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class StefanMaxwell(pybamm.BaseModel):
    """A class that generates the expression tree for Stefan-Maxwell Diffusion in the
    electrolyte.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        The electrolyte concentration
    epsilon : :class:`pybamm.Symbol`
        The (electrolyte/liquid phase) porosity (can be Variable or Parameter)
    j : :class:`pybamm.Symbol`
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, c_e, epsilon, j, param):
        super().__init__()

        # Flux
        N_e = -(epsilon ** param.b) * pybamm.grad(c_e)
        # Porosity change
        deps_dt = -param.beta_surf * j

        # Model
        self.rhs = {
            c_e: (1 / epsilon)
            * (
                -pybamm.div(N_e) / param.C_e
                + param.s / param.gamma_hat_e * j
                - c_e * deps_dt
            )
        }
        self.initial_conditions = {c_e: param.c_e_init}
        self.initial_conditions_ydot = {c_e: 0}
        self.boundary_conditions = {N_e: {"left": 0, "right": 0}}
        self.variables = {"Electrolyte concentration": c_e, "Cation flux": N_e}

        # Cut off if concentration goes negative
        self.events = [pybamm.Function(np.min, c_e)]
