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
            deps_dt = variables["Porosity change"]
        except KeyError:
            epsilon = param.epsilon
            deps_dt = pybamm.Scalar(0)

        # Flux
        N_e = -(epsilon ** param.b) * pybamm.grad(c_e)

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

    def set_leading_order_system(self, c_e, variables):
        param = self.set_of_parameters
        i_cell = param.current_with_time

        # Unpack variables
        j_n = variables["Negative electrode interfacial current density"].orphans[0]
        j_p = variables["Positive electrode interfacial current density"].orphans[0]

        # if porosity is not provided, use the input parameter
        try:
            epsilon = variables["Porosity"]
            deps_n_dt = variables["Negative electrode porosity change"].orphans[0]
            deps_p_dt = variables["Positive electrode porosity change"].orphans[0]
        except KeyError:
            epsilon = param.epsilon
            deps_n_dt = pybamm.Scalar(0)
            deps_p_dt = pybamm.Scalar(0)

        eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]

        # Model
        self.rhs = {
            c_e: 1
            / (param.l_n * eps_n + param.l_s * eps_s + param.l_p * eps_p)
            * (
                (param.s_n - param.s_p) * i_cell
                - c_e * (param.l_n * deps_n_dt + param.l_p * deps_p_dt)
            )
        }
        self.initial_conditions = {c_e: param.c_e_init}

        # Variables
        whole_cell = epsilon.domain
        N_e = pybamm.Broadcast(0, whole_cell)
        self.set_variables(pybamm.Broadcast(c_e, whole_cell), N_e)

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
