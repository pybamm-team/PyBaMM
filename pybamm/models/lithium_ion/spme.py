#
# Single Particle Model with Electrolyte (SPMe)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class SPMe(pybamm.LithiumIonBaseModel):
    """Single Particle Model with Electrolyte (SPMe) of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self):
        super().__init__()

        "Parameters"
        param = pybamm.standard_parameters_lithium_ion

        "Model Variables"
        # Electrolyte concentration (combined leading and first order, nonlinear)
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
        # spatial variables
        spatial_vars = pybamm.standard_spatial_vars

        # surface concentrations
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)

        # open circuit voltage (leading order)
        ocp_n = param.U_n(c_s_n_surf)
        ocp_p = param.U_p(c_s_p_surf)
        ocv = ocp_p - ocp_n

        # exchange current densities (combined leading and first order)
        j0_n = pybamm.interface.exchange_current_density(c_e_n, c_s_n_surf)
        j0_p = pybamm.interface.exchange_current_density(c_e_p, c_s_p_surf)

        # reaction overpotentials (combined leading and first order)
        eta_r_n = pybamm.interface.inverse_butler_volmer(j_n, j0_n, param.ne_n)
        eta_r_p = pybamm.interface.inverse_butler_volmer(j_p, j0_p, param.ne_p)

        # electrode-averaged reaction overpotentials (combined leading and first order)
        eta_r_n_av = pybamm.Integral(eta_r_n, spatial_vars.x_n) / param.l_n
        eta_r_p_av = pybamm.Integral(eta_r_p, spatial_vars.x_p) / param.l_p

        # total reaction overpotential (combined leading and first order)
        eta_r = eta_r_p_av - eta_r_n_av

        # electrolyte potentials (combinded leading and first order)
        # and current (leading order)
        explicit_stefan_maxwell = pybamm.electrolyte_current.explicit_stefan_maxwell
        phi_e, i_e, Delta_Phi_e, eta_c = explicit_stefan_maxwell(
            param, c_e, ocp_n, pybamm.BoundaryValue(eta_r_n, "left")
        )

        # solid phase ohmic losse
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
