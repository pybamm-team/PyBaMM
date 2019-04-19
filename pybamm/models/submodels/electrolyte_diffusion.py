#
# Equation classes for the electrolyte concentration
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class DiffusionModel(pybamm.SubModel):
    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def compute_dimensional_variables(self):
        param = self.set_of_parameters

        c_e = self.variables["Electrolyte concentration"]

        self.variables.update(
            {"Electrolyte concentration [mols m-3]": param.c_e_typ * c_e}
        )


class StefanMaxwell(DiffusionModel):
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

    def __init__(self, set_of_parameters):
        super().__init__()

    def pde_model(self, c_e, j, epsilon=None):
        param = self.set_of_parameters

        # if porosity is not a variable, use the input parameter
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
        self.variables = {"Electrolyte concentration": c_e, "Reduced cation flux": N_e}
        self.compute_dimensional_variables()

        # Cut off if concentration goes negative
        self.events = [pybamm.Function(np.min, c_e)]


class ConstantConcentration(DiffusionModel):
    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

        c_e_n = pybamm.Broadcast(1, domain=["negative electrode"])
        c_e_s = pybamm.Broadcast(1, domain=["separator"])
        c_e_p = pybamm.Broadcast(1, domain=["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        self.variables = {"Electrolyte concentration": c_e}
        self.compute_dimensional_variables()
