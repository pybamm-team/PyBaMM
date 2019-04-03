#
# Single Particle Model (SPM)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class SPM(pybamm.LithiumIonBaseModel):
    """Single Particle Model (SPM) of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self):
        super().__init__()

        "Parameters"
        param = pybamm.standard_parameters_lithium_ion

        "Model Variables"
        # Particle concentration
        c_s_n = pybamm.Variable(
            "Negative particle concentration", domain="negative particle"
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration", domain="positive particle"
        )

        "Submodels"
        # Interfacial current density
        j_n = pybamm.interface.homogeneous_reaction(["negative electrode"])
        j_p = pybamm.interface.homogeneous_reaction(["positive electrode"])

        # Particle models
        negative_particle_model = pybamm.particle.Standard(c_s_n, j_n, param)
        positive_particle_model = pybamm.particle.Standard(c_s_p, j_p, param)

        "Combine Submodels"
        self.update(negative_particle_model, positive_particle_model)

        "Additional Conditions"
        additional_bcs = {}
        self._boundary_conditions.update(additional_bcs)

        "Additional Useful Variables"
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
        j0_n = (1 / param.C_r_n) * c_s_n_surf ** 0.5 * (1 - c_s_n_surf) ** 0.5
        j0_p = (
            (param.gamma_p / param.C_r_p) * c_s_p_surf ** 0.5 * (1 - c_s_p_surf) ** 0.5
        )
        eta_r_n = -2 * pybamm.Function(np.arcsinh, i_cell / (j0_p * param.l_p))
        eta_r_p = -2 * pybamm.Function(np.arcsinh, i_cell / (j0_n * param.l_n))
        eta_r = eta_r_n + eta_r_p

        # electrolyte concentration
        c_e = pybamm.Scalar(1)

        # electrolyte potential
        phi_e = -ocp_n - eta_r_n

        # terminal voltage
        v = ocv + eta_r

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
            "Terminal voltage": v,
            "Electrolyte Potential": phi_e,
            "Electrolyte concentration": c_e,
        }

        self._variables.update(additional_variables)

        "Termination Conditions"
        # Cut-off if either concentration goes negative
        self.events = [pybamm.Function(np.min, c_s_n), pybamm.Function(np.min, c_s_p)]
