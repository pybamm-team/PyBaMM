#
# Doyle-Fuller-Newman (DFN) Model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import numpy as np


class DFN(pybamm.LithiumIonBaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self):
        super().__init__()

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lithium_ion

        "-----------------------------------------------------------------------------"
        "Model Variables"
        # Electrolyte concentration
        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration", ["negative electrode"]
        )
        c_e_s = pybamm.Variable("Separator electrolyte concentration", ["separator"])
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration", ["positive electrode"]
        )
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        # Electrolyte Potential
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential", ["negative electrode"]
        )
        phi_e_s = pybamm.Variable("Separator electrolyte potential", ["separator"])
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential", ["positive electrode"]
        )
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Electrode Potential
        phi_s_n = pybamm.Variable(
            "Negative electrode potential", ["negative electrode"]
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential", ["positive electrode"]
        )

        # Particle concentration
        c_s_n = pybamm.Variable(
            "Negative particle concentration", ["negative particle"]
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration", ["positive particle"]
        )

        "-----------------------------------------------------------------------------"
        "Submodels"
        # Interfacial current density
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)
        j_n = pybamm.interface.butler_volmer(
            param, c_e_n, phi_s_n - phi_e_n, c_s_k_surf=c_s_n_surf
        )
        j_s = pybamm.Broadcast(0, ["separator"])
        j_p = pybamm.interface.butler_volmer(
            param, c_e_p, phi_s_p - phi_e_p, c_s_k_surf=c_s_p_surf
        )
        j = pybamm.Concatenation(j_n, j_s, j_p)

        # Electrolyte models
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(
            c_e, j, param
        )
        electrolyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
            c_e, phi_e, j, param
        )

        # Electrode models
        negative_electrode_current_model = pybamm.electrode.Ohm(phi_s_n, j_n, param)
        positive_electrode_current_model = pybamm.electrode.Ohm(phi_s_p, j_p, param)

        # Particle models
        negative_particle_model = pybamm.particle.Standard(c_s_n, j_n, param)
        positive_particle_model = pybamm.particle.Standard(c_s_p, j_p, param)

        "-----------------------------------------------------------------------------"
        "Combine Submodels"
        self.update(
            negative_particle_model,
            positive_particle_model,
            electrolyte_diffusion_model,
            electrolyte_current_model,
            negative_electrode_current_model,
            positive_electrode_current_model,
        )

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # spatial variables
        spatial_vars = pybamm.standard_spatial_vars

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

        ocv_av = ocp_p_av - ocp_n_av

        ocp_n_left = pybamm.BoundaryValue(ocp_n, "left")
        ocp_p_right = pybamm.BoundaryValue(ocp_p, "right")
        ocv = ocp_p_right - ocp_n_left

        # average elecrolyte overpotential (ohmic + concentration overpotential)
        phi_e_n_av = pybamm.Integral(phi_e_n, spatial_vars.x_n) / param.l_n
        phi_e_p_av = pybamm.Integral(phi_e_p, spatial_vars.x_p) / param.l_p
        eta_e_av = phi_e_p_av - phi_e_n_av

        # solid phase ohmic losses
        Delta_Phi_s_n = phi_s_n - pybamm.BoundaryValue(phi_s_n, "left")
        Delta_Phi_s_n_av = pybamm.Integral(Delta_Phi_s_n, spatial_vars.x_n) / param.l_n

        Delta_Phi_s_p = phi_s_p - pybamm.BoundaryValue(phi_s_p, "right")
        Delta_Phi_s_p_av = pybamm.Integral(Delta_Phi_s_p, spatial_vars.x_p) / param.l_p

        Delta_Phi_s_av = Delta_Phi_s_p_av - Delta_Phi_s_n_av

        # terminal voltage
        v = ocv_av + eta_r_av + eta_e_av + Delta_Phi_s_av

        "-----------------------------------------------------------------------------"
        "Standard Output Variables"

        # Current
        self._variables.update(
            {
                "Total current density": param.current_with_time,
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
        self._variables.update({})

        # Potential
        self._variables.update({})

        "-----------------------------------------------------------------------------"
        "Additional Model Variables"

        self._variables.update({})

        "-----------------------------------------------------------------------------"
        "Defaults and Solver Conditions"
        # default geometry
        self.default_geometry = pybamm.Geometry("1D macro", "1+1D micro")
        # Default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()
        # Cut-off if either concentration goes negative
        self.events = [pybamm.Function(np.min, c_s_n), pybamm.Function(np.min, c_s_p)]
