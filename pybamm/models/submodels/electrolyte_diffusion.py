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
    j : :class:`pybamm.Symbol`
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface
    param : parameter class
        The parameters to use for this submodel
    epsilon : :class:`pybamm.Symbol`
        The (electrolyte/liquid phase) porosity  (only supply if a variable)

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, c_e, j, param, epsilon=None):
        super().__init__()

        # set porosity to parameter if its not supplied
        # as a variable
        if epsilon is None:
            epsilon = param.epsilon

        # Flux
        N_e = -(epsilon ** param.b) * pybamm.grad(c_e)

        # porosity change (note beta_surf must be 0
        # if epsilon is not supplied as a variable)
        deps_dt = -param.beta_surf * j
        # Model
        self.rhs = {
            c_e: (1 / epsilon)
            * (
                -pybamm.div(N_e) / param.C_e
                + param.s / param.gamma_e * j
                - c_e * deps_dt
            )
        }

        self.initial_conditions = {c_e: param.c_e_init}
        self.boundary_conditions = {N_e: {"left": 0, "right": 0}}
        self.variables = {"Electrolyte concentration": c_e, "Cation flux": N_e}

        # Cut off if concentration goes negative
        self.events = [pybamm.Function(np.min, c_e)]
