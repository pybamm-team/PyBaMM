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

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lithium_ion

        "-----------------------------------------------------------------------------"
        "Model Variables"

        # Particle concentration
        c_s_n = pybamm.Variable(
            "Negative particle concentration", domain="negative particle"
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration", domain="positive particle"
        )

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Interfacial current density
        j_n = param.current_with_time / param.l_n
        j_p = -param.current_with_time / param.l_p

        # Particle models
        negative_particle_model = pybamm.particle.Standard(c_s_n, j_n, param)
        positive_particle_model = pybamm.particle.Standard(c_s_p, j_p, param)

        "-----------------------------------------------------------------------------"
        "Combine Submodels"

        self.update(negative_particle_model, positive_particle_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # spatial variables
        spatial_vars = pybamm.standard_spatial_vars

        # electrolyte concentration
        c_e_n = pybamm.Broadcast(1, domain=["negative electrode"])
        c_e_s = pybamm.Broadcast(1, domain=["separator"])
        c_e_p = pybamm.Broadcast(1, domain=["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        # interfacial current density
        j_n = pybamm.Broadcast(j_n, ["negative electrode"])
        j_s = pybamm.Broadcast(pybamm.Scalar(0), domain=["separator"])
        j_p = pybamm.Broadcast(j_p, ["positive electrode"])
        j = pybamm.Concatenation(j_n, j_s, j_p)

        # exhange current density
        j0_n = pybamm.interface.exchange_current_density(c_e_n, pybamm.surf(c_s_n))
        j0_s = pybamm.Broadcast(pybamm.Scalar(0), domain=["separator"])
        j0_p = pybamm.interface.exchange_current_density(c_e_p, pybamm.surf(c_s_p))
        j0 = pybamm.Concatenation(j0_n, j0_s, j0_p)

        # reaction overpotentials
        eta_r_n = pybamm.interface.inverse_butler_volmer(j_n, j0_n, param.ne_n)
        eta_r_p = pybamm.interface.inverse_butler_volmer(j_p, j0_p, param.ne_p)
        eta_r_n_av = pybamm.Integral(eta_r_n, spatial_vars.x_n) / param.l_n
        eta_r_p_av = pybamm.Integral(eta_r_p, spatial_vars.x_p) / param.l_p
        eta_r_av = eta_r_p_av - eta_r_n_av

        # open circuit voltage
        ocp_n = pybamm.Broadcast(param.U_n(pybamm.surf(c_s_n)), ["negative electrode"])
        ocp_p = pybamm.Broadcast(param.U_p(pybamm.surf(c_s_p)), ["positive electrode"])
        ocp_n_av = pybamm.Integral(ocp_n, spatial_vars.x_n) / param.l_n
        ocp_p_av = pybamm.Integral(ocp_p, spatial_vars.x_p) / param.l_p
        ocp_n_left = pybamm.BoundaryValue(ocp_n, "left")
        ocp_p_right = pybamm.BoundaryValue(ocp_p, "right")
        ocv_av = ocp_p_av - ocp_n_av
        ocv = ocp_p_right - ocp_n_left

        # electrolyte potential, current, ohmic losses, and concentration overpotential
        elosm = pybamm.electrolyte_current.explicit_leading_order_stefan_maxwell
        phi_e, i_e, Delta_Phi_e_av, eta_c_av = elosm(param, c_e, ocp_n, eta_r_n)
        eta_e_av = eta_c_av + Delta_Phi_e_av

        # electrode potentials, current, and solid phase ohmic losses
        eloo = pybamm.electrode.explicit_leading_order_ohm
        phi_s, i_s, Delta_Phi_s_av = eloo(param, phi_e, ocp_p, eta_r_p)
        phi_s_n = phi_s.orphans[0]
        phi_s_p = phi_s.orphans[2]
        i_s_n = i_s.orphans[0]
        i_s_p = i_s.orphans[2]

        # terminal voltage
        v = ocv_av + eta_r_av

        "-----------------------------------------------------------------------------"
        "Standard Output Variables"

        # Current
        self._variables.update(
            {
                "Total current density": param.current_with_time,
                "Negative electrode current density": i_s_n,
                "Positive electrode current density": i_s_p,
                "Electrolyte current density": i_e,
                "Interfacial current density": j,
                "Exchange current density": j0,
            }
        )

        # Voltage
        self._variables.update(
            {
                "Negative electrode open circuit potential": ocp_n,
                "Positive electrode open circuit potential": ocp_p,
                "Average negative electrode open circuit potential": ocp_n_av,
                "Average positive electrode open circuit potential": ocp_p_av,
                "Average open circuit voltage": ocv_av,
                "Measured open circuit voltage": ocv,
                "Terminal voltage": v,
            }
        )

        # Overpotential
        self._variables.update(
            {
                "Negative reaction overpotential": eta_r_n,
                "Positive reaction overpotential": eta_r_p,
                "Average negative reaction overpotential": eta_r_n_av,
                "Average positive reaction overpotential": eta_r_p_av,
                "Average reaction overpotential": eta_r_av,
                "Average electrolyte overpotential": eta_e_av,
                "Average solid phase ohmic losses": Delta_Phi_s_av,
            }
        )

        # Concentration
        self._variables.update({"Electrolyte concentration": c_e})

        # Potential
        self._variables.update(
            {
                "Negative electrode potential": phi_s_n,
                "Positive electrode potential": phi_s_p,
                "Electrolyte potential": phi_e,
            }
        )

        "-----------------------------------------------------------------------------"
        "Additional Output Variables"

        self._variables.update(
            {
                "Average concentration overpotential": eta_c_av,
                "Average electrolyte ohmic losses": Delta_Phi_e_av,
            }
        )

        "-----------------------------------------------------------------------------"
        "Defaults and Solver Conditions"
        # default geometry
        self.default_geometry = pybamm.Geometry("1D macro", "1D micro")

        # Cut-off if either concentration goes negative
        self.events = [pybamm.Function(np.min, c_s_n), pybamm.Function(np.min, c_s_p)]
