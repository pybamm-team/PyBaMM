#
# Equation classes for the electrolyte concentration
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class ElectrolyteDiffusionModel(pybamm.SubModel):
    """Base model class for diffusion in the electrolyte.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_variables(self, c_e, N_e):
        param = self.set_of_parameters

        self.variables = {
            "Electrolyte concentration": c_e,
            "Reduced cation flux": N_e,
            "Electrolyte concentration [mols m-3]": param.c_e_typ * c_e,
        }


class StefanMaxwell(ElectrolyteDiffusionModel):
    """A class that generates the expression tree for Stefan-Maxwell Diffusion in the
    electrolyte.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`ElectrolyteDiffusionModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_differential_system(self, c_e, variables):
        param = self.set_of_parameters

        # Unpack variables
        j = variables["Interfacial current density"]

        # if porosity is not provided, use the input parameter
        try:
            epsilon = variables["Porosity"]
        except KeyError:
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
        self.set_variables(c_e, N_e)

        # Cut off if concentration goes negative
        self.events = [pybamm.Function(np.min, c_e)]


class ConstantConcentration(ElectrolyteDiffusionModel):
    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

        c_e_n = pybamm.Broadcast(1, domain=["negative electrode"])
        c_e_s = pybamm.Broadcast(1, domain=["separator"])
        c_e_p = pybamm.Broadcast(1, domain=["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        N_e = pybamm.Broadcast(0, c_e.domain)

        self.set_variables(c_e, N_e)
