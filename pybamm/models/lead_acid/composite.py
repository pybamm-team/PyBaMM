#
# Lead-acid Composite model
#
import pybamm


class Composite(pybamm.LeadAcidBaseModel):
    """Composite model for lead-acid, from [1]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.

    **Extends:** :class:`pybamm.LeadAcidBaseModel`
    """

    def __init__(self, options=None):
        # Update own model with submodels
        super().__init__(options)
        self.name = "Composite model"

        # Leading order model and variables
        leading_order_model = pybamm.lead_acid.LOQS(options)
        self.update(leading_order_model)

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid
        self._set_of_parameters = param

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.standard_variables.c_e
        self.variables["Electrolyte concentration"] = c_e

        "-----------------------------------------------------------------------------"
        "Boundary conditions"

        self.set_boundary_conditions(None)

        "-----------------------------------------------------------------------------"
        "Submodels"
        j_n_0 = self.variables["Negative electrode interfacial current density"]
        j_p_0 = self.variables["Positive electrode interfacial current density"]

        # Exchange-current density
        int_curr_model = pybamm.interface.LeadAcidReaction(param)
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        c_e_n, _, c_e_p = c_e.orphans
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p, pos)

        # Open-circuit potentials
        ocp_n = param.U_n(c_e_n)
        ocp_p = param.U_p(c_e_p)

        # Electrolyte concentration
        reactions = {
            "main": {
                "neg": {"s_plus": param.s_n, "aj": j_n_0},
                "pos": {"s_plus": param.s_p, "aj": j_p_0},
                "porosity change": leading_order_model.variables["Porosity change"],
            }
        }
        electrolyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        electrolyte_conc_model.set_differential_system(self.variables, reactions)

        self.update(electrolyte_conc_model)

        # Electrolyte current model
        self.set_electrolyte_current_model(int_curr_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Interfacial current density
        j_vars = int_curr_model.get_derived_interfacial_currents(
            j_n_0, j_p_0, j0_n, j0_p
        )
        self.variables.update(j_vars)

        # Potentials
        eta_r_n = int_curr_model.get_inverse_butler_volmer(j_n_0, j0_n, neg)
        eta_r_p = int_curr_model.get_inverse_butler_volmer(j_p_0, j0_p, pos)
        pot_model = pybamm.potential.Potential(param)
        ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n, ocp_p)
        eta_r_vars = pot_model.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
        self.variables.update({**ocp_vars, **eta_r_vars})

        # Load electrolyte and electrode potentials
        electrode_model = pybamm.electrode.Ohm(param)
        electrolyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
            param
        )

        # Negative electrode potential
        phi_s_n = electrode_model.get_neg_pot_explicit_combined(self.variables)
        self.variables["Negative electrode potential"] = phi_s_n

        # Electrolyte potential
        electrolyte_vars = electrolyte_current_model.get_explicit_combined(
            self.variables
        )
        self.variables.update(electrolyte_vars)

        # Electrode
        electrode_vars = electrode_model.get_explicit_combined(self.variables)
        self.variables.update(electrode_vars)

        # Cut-off voltage
        voltage = self.variables["Terminal voltage"]
        self.events.append(voltage - param.voltage_low_cut)

    def set_boundary_conditions(self, bc_variables):
        """Set boundary conditions, dependent on self.options"""
        param = self.set_of_parameters
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality == 1:
            current_bc = param.current_with_time
            self.variables["Current collector current density"] = current_bc

    def set_electrolyte_current_model(self, int_curr_model):
        if self.options["capacitance"] is False:
            return

        param = self.set_of_parameters
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        delta_phi_n_av = pybamm.Variable(
            "Average neg electrode surface potential difference"
        )
        delta_phi_p_av = pybamm.Variable(
            "Average pos electrode surface potential difference"
        )

        # Average composite interfacial current density
        c_e = self.variables["Electrolyte concentration"]
        c_e_n, _, c_e_p = c_e.orphans
        c_e_n_av = pybamm.average(c_e_n)
        c_e_p_av = pybamm.average(c_e_p)
        ocp_n_av = param.U_n(c_e_n_av)
        ocp_p_av = param.U_p(c_e_p_av)
        eta_r_n_av = delta_phi_n_av - ocp_n_av
        eta_r_p_av = delta_phi_p_av - ocp_p_av
        j0_n_av = int_curr_model.get_exchange_current_densities(c_e_n_av, neg)
        j0_p_av = int_curr_model.get_exchange_current_densities(c_e_p_av, pos)
        j_n_av = int_curr_model.get_butler_volmer(j0_n_av, eta_r_n_av, neg)
        j_p_av = int_curr_model.get_butler_volmer(j0_p_av, eta_r_p_av, pos)

        # Make dictionaries to pass to submodel
        i_boundary_cc = self.variables["Current collector current density"]
        variables_av = {
            "Negative electrode surface potential difference": delta_phi_n_av,
            "Positive electrode surface potential difference": delta_phi_p_av,
            "Current collector current density": i_boundary_cc,
            "Electrolyte concentration": c_e,
            "Porosity": self.variables["Porosity"],
        }
        reactions_av = {"main": {"neg": {"aj": j_n_av}, "pos": {"aj": j_p_av}}}

        # Call submodel
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesCapacitance(
            param, self.options["capacitance"]
        )
        eleclyte_current_model.set_leading_order_system(variables_av, reactions_av, neg)
        eleclyte_current_model.set_leading_order_system(variables_av, reactions_av, pos)
        self.update(eleclyte_current_model)

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        if self.options["capacitance"] == "algebraic":
            return pybamm.ScikitsDaeSolver()
        else:
            return pybamm.ScikitsOdeSolver()
