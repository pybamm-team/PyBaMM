#
# Lead-acid Newman-Tiedemann model
#
import pybamm


class NewmanTiedemann(pybamm.LeadAcidBaseModel):
    """Porous electrode model for lead-acid, from [1]_.

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: I. Physical Model.
           arXiv preprint arXiv:1902.01771, 2019.

    **Extends:** :class:`pybamm.LeadAcidBaseModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "Newman-Tiedemann model"

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid
        capacitance_options = self.options["capacitance"]

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e, eps, delta_phi_n, delta_phi_p, potentials = self.get_model_variables()

        "-----------------------------------------------------------------------------"
        "Submodels"
        # Exchange-current density
        c_e_n, _, c_e_p = c_e.orphans
        int_curr_model = pybamm.interface.LeadAcidReaction(param)
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p)

        # Potentials
        ocp_n = param.U_n(c_e_n)
        ocp_p = param.U_p(c_e_p)
        eta_r_n = delta_phi_n - ocp_n
        eta_r_p = delta_phi_p - ocp_p

        # Interfacial current density
        j_n = int_curr_model.get_butler_volmer(j0_n, eta_r_n)
        j_p = int_curr_model.get_butler_volmer(j0_p, eta_r_p)

        # Porosity
        porosity_model = pybamm.porosity.Standard(param)
        porosity_model.set_differential_system(eps, j_n, j_p)

        # Electrolyte concentration
        reactions = {
            "main": {
                "neg": {"s_plus": param.s_n, "aj": j_n},
                "pos": {"s_plus": param.s_p, "aj": j_p},
                "porosity change": porosity_model.variables["Porosity change"],
            }
        }
        # Electrolyte diffusion model
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        electrolyte_diffusion_model.set_differential_system(c_e, reactions, eps)

        self.update(porosity_model, electrolyte_diffusion_model)

        if self.options["capacitance"] is False:
            phi_e, phi_s_n, phi_s_p = potentials
            eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
                param
            )
            eleclyte_current_model.set_algebraic_system(phi_e, c_e, reactions, eps)

            # Electrode models
            eps_n, _, eps_p = eps.orphans
            negative_electrode_current_model = pybamm.electrode.Ohm(param)
            negative_electrode_current_model.set_algebraic_system(
                phi_s_n, reactions, eps_n
            )
            positive_electrode_current_model = pybamm.electrode.Ohm(param)
            positive_electrode_current_model.set_algebraic_system(
                phi_s_p, reactions, eps_p
            )
            self.update(
                eleclyte_current_model,
                negative_electrode_current_model,
                positive_electrode_current_model,
            )
        else:
            # Electrolyte current
            eps_n, _, eps_p = eps.orphans
            eleclyte_current_model = pybamm.electrolyte_current.MacInnesCapacitance(
                param, capacitance_options
            )
            eleclyte_current_model.set_full_system(delta_phi_n, c_e_n, reactions, eps_n)
            eleclyte_current_model.set_full_system(delta_phi_p, c_e_p, reactions, eps_p)

            # Post-process electrolyte model
            eleclyte_current_model.set_post_processed(c_e, eps)
            self.update(eleclyte_current_model)

        "-----------------------------------------------------------------------------"
        "Post-process"

        # Exchange-current density
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

        # Potentials
        pot_model = pybamm.potential.Potential(param)
        ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n, ocp_p)
        eta_r_vars = pot_model.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
        self.variables.update({**ocp_vars, **eta_r_vars})

        # Voltage
        if capacitance_options is False:
            phi_s_n = self.variables["Negative electrode potential"]
            phi_s_p = self.variables["Positive electrode potential"]
            i_s_n = self.variables["Negative electrode current density"]
            i_s_p = self.variables["Positive electrode current density"]
            volt_vars = positive_electrode_current_model.get_variables(
                phi_s_n, phi_s_p, i_s_n, i_s_p
            )
            self.variables.update(volt_vars)

        # Cut-off voltage
        voltage = pybamm.BoundaryValue(delta_phi_p, "right") - pybamm.BoundaryValue(
            delta_phi_n, "left"
        )
        self.events.append(voltage - param.voltage_low_cut)

    def get_model_variables(self):
        c_e = pybamm.standard_variables.c_e
        eps = pybamm.standard_variables.eps

        if self.options["capacitance"] is False:
            phi_e = pybamm.standard_variables.phi_e
            phi_s_p = pybamm.standard_variables.phi_s_p
            phi_s_n = pybamm.standard_variables.phi_s_n
            delta_phi_n = phi_s_n - phi_e.orphans[0]
            delta_phi_p = phi_s_p - phi_e.orphans[2]
            potentials = (phi_e, phi_s_n, phi_s_p)
        else:
            delta_phi_n = pybamm.standard_variables.delta_phi_n
            delta_phi_p = pybamm.standard_variables.delta_phi_p
            potentials = None

        return c_e, eps, delta_phi_n, delta_phi_p, potentials

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        if self.options["capacitance"] == "differential":
            return pybamm.ScikitsOdeSolver()
        else:
            return pybamm.ScikitsDaeSolver()
