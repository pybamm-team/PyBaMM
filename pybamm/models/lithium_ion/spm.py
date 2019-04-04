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
        # j_n = pybamm.interface.homogeneous_reaction(["negative electrode"])
        # j_p = pybamm.interface.homogeneous_reaction(["positive electrode"])
        j_n = param.current_with_time / param.l_n
        j_p = -param.current_with_time / param.l_p

        # Particle models
        negative_particle_model = pybamm.particle.Standard(c_s_n, j_n, param)
        positive_particle_model = pybamm.particle.Standard(c_s_p, j_p, param)

        "Combine Submodels"
        self.update(negative_particle_model, positive_particle_model)

        "Additional Conditions"
        additional_bcs = {}
        self._boundary_conditions.update(additional_bcs)

        "Additional Useful Variables"
        # surface concentrations
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)

        # electrolyte concentration
        c_e_n = pybamm.Scalar(1)
        c_e_p = pybamm.Scalar(1)
        c_e = pybamm.Scalar(1)

        # open circuit voltage
        ocp_n = param.U_n(c_s_n_surf)
        ocp_p = param.U_p(c_s_p_surf)
        ocv = ocp_p - ocp_n

        # exhange current density
        j0_n = pybamm.interface.exchange_current_density(
            c_e_n, c_s_n_surf, ["negative electrode"]
        )
        j0_p = pybamm.interface.exchange_current_density(
            c_e_p, c_s_p_surf, ["positive electrode"]
        )

        # reaction overpotentials
        eta_r_n = pybamm.interface.inverse_butler_volmer(j_n, j0_n, param.ne_n)
        eta_r_p = pybamm.interface.inverse_butler_volmer(j_p, j0_p, param.ne_p)
        eta_r = eta_r_p - eta_r_n

        # electrolyte potential
        phi_e = -ocp_n - eta_r_n

        # terminal voltage
        v = ocv + eta_r

        additional_variables = {
            "current": param.current_with_time,
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
