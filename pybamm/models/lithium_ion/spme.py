#
# Single Particle Model with Electrolyte (SPMe)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class SPMe(pybamm.BaseModel):
    """Single Particle Model with Electrolyte (SPMe) of a lithium-ion battery.

    Attributes
    ----------

    rhs: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the rhs
    initial_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions
    boundary_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the boundary conditions
    variables: dict
        A dictionary that maps strings to expressions that represent
        the useful variables

    """

    def __init__(self):
        super().__init__()

        "Parameters"
        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lithium_ion.__dict__)

        "Model Variables"
        # Electrolyte concentration
        c_e_n = pybamm.Variable("c_e_n", ["negative electrode"])
        c_e_s = pybamm.Variable("c_e_s", ["separator"])
        c_e_p = pybamm.Variable("c_e_p", ["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        # Particle concentration
        c_s_n = pybamm.Variable("c_s_n", ["negative particle"])
        c_s_p = pybamm.Variable("c_s_p", ["positive particle"])

        "Submodels"
        # Interfacial current density
        j_n = pybamm.interface.homogeneous_reaction(["negative electrode"])
        j_p = pybamm.interface.homogeneous_reaction(["positive electrode"])
        j = pybamm.interface.homogeneous_reaction(
            ["negative electrode", "separator", "positive electrode"]
        )

        # Particle models
        negative_particle_model = pybamm.particle.Standard(c_s_n, j_n, param)
        positive_particle_model = pybamm.particle.Standard(c_s_p, j_p, param)

        # Electrolyte models
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(
            c_e, j, param
        )

        "Combine Submodels"
        self.update(
            negative_particle_model,
            positive_particle_model,
            electrolyte_diffusion_model,
        )

        "Additional Conditions"
        additional_bcs = {}
        self._boundary_conditions.update(additional_bcs)

        "Additional Model Variables"
        # current
        i_cell = param.current_with_time

        # surface concentrations
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)

        # open circuit voltage
        ocp_n = param.U_n(c_s_n_surf)
        ocp_p = param.U_p(c_s_p_surf)
        ocv = ocp_p - ocp_n

        # reaction overpotentials
        # j0_n = (
        #     param.m_n
        #     * c_s_n_surf ** 0.5
        #     * (1 - c_s_n_surf) ** 0.5
        #     * (1 + param.C_e * c_e_n) ** 0.5
        # )
        # j0_p = (
        #     param.m_p
        #     * param.gamma_p
        #     * c_s_p_surf ** 0.5
        #     * (1 - c_s_p_surf) ** 0.5
        #     * (1 + param.C_e * c_e_p) ** 0.5
        # )

        # TODO: electrode average j0_n, j0_p
        j0_n_av = pybamm.Scalar(1) + 0
        j0_p_av = pybamm.Scalar(1) + 0

        eta_r_n = -2 * pybamm.Function(np.arcsinh, i_cell / (j0_p_av * param.l_p))
        eta_r_p = -2 * pybamm.Function(np.arcsinh, i_cell / (j0_n_av * param.l_n))
        eta_r = eta_r_n + eta_r_p

        # electrolyte potentials
        # TODO: add the expressions for these

        # TODO: electrode average c_e_p and c_e_n
        c_e_p_av = pybamm.Scalar(0)
        c_e_n_av = pybamm.Scalar(0)

        # combined electrolyte concentrations
        c_e_n_combined = 1 + param.C_e * c_e_n
        c_e_s_combined = 1 + param.C_e * c_e_s
        c_e_p_combined = 1 + param.C_e * c_e_p
        c_e_combined = 1 + param.C_e * c_e

        # concentration overpotential
        eta_c = 2 * param.C_e * (1 - param.t_plus) * (c_e_p_av - c_e_n_av)

        # electrolyte ohmic losses
        Delta_Phi_elec = (
            -param.C_e
            * i_cell
            * (1 / param.gamma_e)
            / param.kappa_e(c_e)
            * (
                param.l_n / (3 * param.epsilon_n ** param.b)
                + param.l_s / (param.epsilon_s ** param.b)
                + param.l_p / (3 * param.epsilon_p ** param.b)
            )
        )

        # solid phase ohmic losses
        Delta_Phi_solid = (
            -i_cell / 3 * (param.l_p / param.sigma_p + param.l_n / param.sigma_n)
        )

        # terminal voltage
        v = ocv + eta_r + eta_c + Delta_Phi_elec + Delta_Phi_solid

        additional_variables = {
            "current": i_cell,
            "Negative interfacial current density": j_n,
            "Positive interfacial current density": j_p,
            "Negative electrode open circuit potential": ocp_n,
            "Positive electrode open circuit potential": ocp_p,
            "Open circuit voltage": ocv,
            "Negative reaction overpotential": eta_r_n,
            "Positive reaction overpotential": eta_r_p,
            "Reaction overpotential": eta_r,
            "Concentration overpotential": eta_c,
            "Electrolyte ohmic losses": Delta_Phi_elec,
            "Solid phase ohmic losses": Delta_Phi_solid,
            "Terminal voltage": v,
            "Leading order electrolyte concentration": pybamm.Scalar(0),
            "First order negative electrolyte concentration": c_e_n,
            "First order separator electrolyte concentration": c_e_s,
            "First order positive electrolyte concentration": c_e_p,
            "Negative electrolyte concentration": c_e_n_combined,
            "Separator electrolyte concentration": c_e_s_combined,
            "Positive electrolyte concentration": c_e_p_combined,
            "Electrolyte concentration": c_e_combined,
        }
        self._variables.update(additional_variables)

        "Termination Conditions"
        # Cut-off if either concentration goes negative
        self.events = [pybamm.Function(np.min, c_s_n), pybamm.Function(np.min, c_s_p)]
