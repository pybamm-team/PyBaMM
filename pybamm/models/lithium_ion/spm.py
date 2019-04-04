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
        "Additional Conditions"
        additional_bcs = {}
        self._boundary_conditions.update(additional_bcs)

        "-----------------------------------------------------------------------------"
        "Post-Processing"
        # electrolyte concentration
        c_e = pybamm.Scalar(1)

        # exhange current density
        j0_n = pybamm.interface.exchange_current_density(
            c_e, pybamm.surf(c_s_n), ["negative electrode"]
        )
        j0_s = pybamm.Scalar(0, ["separator"])
        j0_p = pybamm.interface.exchange_current_density(
            c_e, pybamm.surf(c_s_p), ["positive electrode"]
        )
        j0 = pybamm.Concatenation(j0_n, j0_s, j0_p)

        # reaction overpotentials
        eta_r_n = pybamm.interface.inverse_butler_volmer(j_n, j0_n, param.ne_n)
        eta_r_p = pybamm.interface.inverse_butler_volmer(j_p, j0_p, param.ne_p)
        eta_r = eta_r_p - eta_r_n

        # open circuit voltage
        ocp_n = param.U_n(pybamm.surf(c_s_n))
        ocp_p = param.U_p(pybamm.surf(c_s_p))
        ocp_n_left = ocp_n
        ocp_p_right = ocp_p
        ocv = ocp_p_right - ocp_n_left

        # electrolyte potential
        phi_e = -ocp_n - eta_r_n

        # electrolyte potential, current, ohmic losses, and concentration overpotential

        # electrode potentials, current, and solid phase ohmic losses
        phi_s, i_s, Delta_Phi_s = pybamm.electrode.explicit_solution_ohm(
            param, phi_e, ocp_p, eta_r_p
        )

        # terminal voltage
        v = ocv + eta_r

        "-----------------------------------------------------------------------------"
        "Standard Output Variables"
        # A standard set of output variables for each type of variable:
        # concentrations, potentials, currents, voltages, and overpotentials is
        # included in the comments below. Some output variables are already included
        # in the output variable dict within submodels. We use different comment styles
        # to indicate which variables need to be inluded and which don't (note that
        # what is already included varies from model to model).  Variables which need
        # to still be included are commented using:

        "- variable still to be included"

        # and those which have already been included are commented using:

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
        " - Open circuit voltage"
        " - Terminal voltage"

        self._variables.update(
            {
                "Negative electrode open circuit potential": ocp_n,
                "Positive electrode open circuit potential": ocp_n,
                "Left-most negative electrode open circuit potential": ocp_n_left,
                "Right-most positive electrode open circuit potential": ocp_p_right,
                "Open circuit voltage": ocv,
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
        # Standard flux outputs:
        #
        # - Negative particle flux
        # - Positive particle flux
        " - Electrolyte flux"

        self._variables.update({"Electrolyte flux": N_e})

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
