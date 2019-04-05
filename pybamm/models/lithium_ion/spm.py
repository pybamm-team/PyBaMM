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

        # electrode potentials, current, and solid phase ohmic losses
        eloo = pybamm.electrode.explicit_leading_order_ohm
        phi_s, i_s, Delta_Phi_s_av = eloo(param, phi_e, ocp_p, eta_r_p)

        # terminal voltage
        v = ocv_av + eta_r_av

        "-----------------------------------------------------------------------------"
        "Standard Output Variables"
        # A standard set of output variables for each type of variable:
        # concentrations, potentials, currents, voltages, and overpotentials is
        # included in the comments below. Some output variables are already included
        # in the output variable dict within submodels. We use different comment styles
        # to indicate which variables need to be inluded and which don't (note that
        # what is already included varies from model to model).  Variables which need
        # to be included now are commented using:

        "- variable still to be included"

        # and those which have already been included in submodels are commented using:

        # variable that is included by a submodel

        # Please copy this standard output set along with this explanation into new
        # models to ensure consistent outputs across models for testing and comparison
        # purposes.

        # -----------------------------------------------------------------------------
        # Standard current outputs:
        #
        " - Total current density"
        " - Electrode current density"
        " - Electrolyte current density"
        " - Interfacial current density"
        " - Exchange current density"

        self._variables.update(
            {
                "Total current density": param.current_with_time,
                "Electrode current density": i_s,
                "Electrolyte current density": i_e,
                "Interfacial current density": j,
                "Exchange current density": j0,
            }
        )

        # -----------------------------------------------------------------------------
        # Standard voltage outputs:
        #
        " - Negative open circuit potential"
        " - Positive open circuit potential"
        " - Average negative open circuit potential"
        " - Average positive open circuit potential"
        " - Average open circuit voltage"
        " - Measured open circuit voltage"
        " - Terminal voltage"

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

        # -----------------------------------------------------------------------------
        # Standard (electrode-averaged) overpotential outputs:
        #
        " - Average negative reaction overpotential"
        " - Average positive reaction overpotential"
        " - Average reaction overpotential"
        " - Average concentration overpotential"
        " - Average electrolyte ohmic losses"
        " - Average solid phase ohmic losses"

        self._variables.update(
            {
                "Average negative reaction overpotential": eta_r_n_av,
                "Average positive reaction overpotential": eta_r_p_av,
                "Average reaction overpotential": eta_r_av,
                "Average concentration overpotential": eta_c_av,
                "Average electrolyte ohmic losses": Delta_Phi_e_av,
                "Average solid phase ohmic losses": Delta_Phi_s_av,
            }
        )

        # -----------------------------------------------------------------------------
        # Standard concentration outputs:
        #
        # - Negative particle concentration
        # - Positive particle concentration
        # - Negative particle surface concentration
        # - Positive particle surface concentration
        " - Electrolyte concentraction"

        self._variables.update({"Electrolyte concentration": c_e})

        # -----------------------------------------------------------------------------
        # Standard potential outputs:
        #
        " - Electrode Potential"
        " - Electrolyte Potential"

        self._variables.update(
            {"Electrode potential": phi_s, "Electrolyte potential": phi_e}
        )

        "-----------------------------------------------------------------------------"
        "Additional Output Variables"
        # Define any other output variables here
        additional_ouput_vars = {}
        self._variables.update(additional_ouput_vars)

        "-----------------------------------------------------------------------------"
        "Termination Conditions"
        # Cut-off if either concentration goes negative
        self.events = [pybamm.Function(np.min, c_s_n), pybamm.Function(np.min, c_s_p)]
