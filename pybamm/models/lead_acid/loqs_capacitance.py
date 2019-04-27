#
# Lead-acid LOQS model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class LOQSCapacitance(pybamm.LeadAcidBaseModel):
    """Leading-Order Quasi-Static model for lead-acid, with capacitance effects

    **Extends**: :class:`pybamm.LeadAcidBaseModel`

    """

    def __init__(self):
        super().__init__()

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.Variable("Electrolyte concentration")
        delta_phi_n = pybamm.Variable("Negative electrode potential difference")
        delta_phi_p = pybamm.Variable("Positive electrode potential difference")
        epsilon = pybamm.standard_variables.eps_piecewise_constant

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Exchange-current density
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        int_curr_model = pybamm.interface.LeadAcidReaction(param)
        j0_n = int_curr_model.get_exchange_current_densities(c_e, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e, pos)

        # Open-circuit potential and reaction overpotential
        ocp_n = param.U_n(c_e)
        ocp_p = param.U_p(c_e)
        eta_r_n = delta_phi_n - ocp_n
        eta_r_p = delta_phi_p - ocp_p

        # Interfacial current density
        j_n = int_curr_model.get_butler_volmer(j0_n, eta_r_n, ["negative electrode"])
        j_p = int_curr_model.get_butler_volmer(j0_p, eta_r_p, ["positive electrode"])

        # Porosity
        porosity_model = pybamm.porosity.Standard(param)
        porosity_model.set_leading_order_system(epsilon, j_n, j_p)

        # Electrolyte concentration
        por_vars = porosity_model.variables
        deps_n_dt = por_vars["Negative electrode porosity change"].orphans[0]
        deps_p_dt = por_vars["Positive electrode porosity change"].orphans[0]
        reactions = {
            "main": {
                "neg": {"s_plus": param.s_n, "aj": j_n, "deps_dt": deps_n_dt},
                "pos": {"s_plus": param.s_p, "aj": j_p, "deps_dt": deps_p_dt},
            }
        }
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_leading_order_system(c_e, reactions, epsilon=epsilon)

        # Electrolyte current
        eleclyte_current_model_n = pybamm.electrolyte_current.MacInnesCapacitance(param)
        eleclyte_current_model_n.set_leading_order_system(delta_phi_n, reactions, neg)
        eleclyte_current_model_p = pybamm.electrolyte_current.MacInnesCapacitance(param)
        eleclyte_current_model_p.set_leading_order_system(delta_phi_p, reactions, pos)
        self.update(
            porosity_model,
            eleclyte_conc_model,
            eleclyte_current_model_n,
            eleclyte_current_model_p,
        )

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Exchange-current density
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

        # Potentials
        pot_model = pybamm.potential.Potential(param)
        ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n, ocp_p)
        eta_r_vars = pot_model.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
        self.variables.update({**ocp_vars, **eta_r_vars})

        # Electrolyte: post-process
        i_s_n = eleclyte_current_model_n.variables[
            "Negative electrolyte current density"
        ]
        i_s_p = eleclyte_current_model_p.variables[
            "Positive electrolyte current density"
        ]
        electrolyte_vars = eleclyte_current_model_p.get_post_processed_leading_order(
            delta_phi_n, i_s_n, i_s_p
        )
        self.variables.update(electrolyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        phi_e = self.variables["Electrolyte potential"]
        electrode_vars = electrode_model.get_explicit_leading_order(
            ocp_p, eta_r_p, phi_e
        )
        self.variables.update(electrode_vars)

        "-----------------------------------------------------------------------------"
        "Default Solver"

        # Use stiff solver
        self.default_solver = pybamm.ScipySolver("BDF")
