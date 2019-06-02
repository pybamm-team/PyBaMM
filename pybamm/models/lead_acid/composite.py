#
# Lead-acid Composite model
#
import pybamm


class Composite(pybamm.LeadAcidBaseModel):
    """Composite model for lead-acid, from [1]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`.

    Notes
    -----
    The composite solution is computed as follows:
    - Get leading-order concentration and porosity from the leading-order model
    - Solve for electrolyte concentration, using leading-order porosity and uniform
      interfacial current density
    - Calculate average first-order surface potential differences
    - Calculate first-order electrolyte and electrode potentials, using average surface
      potential differences
    - Calcualte first-order surface potential-differences and interfacial current
      densities using first-order potentials, and hence update porosity
    - Optionally, post-process to find convection velocity

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
        self.leading_order_variables = leading_order_model.variables

        # Model variables
        self.variables["Electrolyte concentration"] = pybamm.standard_variables.c_e

        # Submodels
        self.set_boundary_conditions(None)
        int_curr_model = pybamm.interface.LeadAcidReaction(self.set_of_parameters)
        self.set_diffusion_submodel()
        self.set_electrolyte_current_model(int_curr_model)
        self.set_current_variables()
        self.set_interface_variables(int_curr_model)
        self.set_convection_variables()

    def set_boundary_conditions(self, bc_variables):
        """Set boundary conditions, dependent on self.options"""
        param = self.set_of_parameters
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality == 0:
            current_bc = param.current_with_time
            self.variables["Current collector current density"] = current_bc

    def set_diffusion_submodel(self):
        param = self.set_of_parameters
        j_n_0 = self.variables["Negative electrode interfacial current density"]
        j_p_0 = self.variables["Positive electrode interfacial current density"]
        self.reactions = {
            "main": {
                "neg": {"s_plus": param.s_n, "aj": j_n_0},
                "pos": {"s_plus": param.s_p, "aj": j_p_0},
                "porosity change": self.variables["Porosity change"],
            }
        }

        electrolyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        electrolyte_conc_model.set_differential_system(self.variables, self.reactions)
        self.update(electrolyte_conc_model)

    def set_electrolyte_current_model(self, int_curr_model):
        param = self.set_of_parameters
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        # Average composite interfacial current density
        i_bnd_cc = self.variables["Current collector current density"]
        c_e = self.variables["Electrolyte concentration"]
        c_e_n, _, c_e_p = c_e.orphans
        c_e_n_av = pybamm.average(c_e_n)
        c_e_p_av = pybamm.average(c_e_p)
        ocp_n_av = param.U_n(c_e_n_av)
        ocp_p_av = param.U_p(c_e_p_av)
        j0_n_av = int_curr_model.get_exchange_current_densities(c_e_n_av, neg)
        j0_p_av = int_curr_model.get_exchange_current_densities(c_e_p_av, pos)

        if self.options["capacitance"] is False:
            eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
                param
            )
            # j_n_av = int_curr_model.get_homogeneous_interfacial_current(i_bnd_cc, neg)
            # j_p_av = int_curr_model.get_homogeneous_interfacial_current(i_bnd_cc, pos)
            # eta_r_n_av = int_curr_model.get_inverse_butler_volmer(j_n_av, j0_n_av, neg)
            # eta_r_p_av = int_curr_model.get_inverse_butler_volmer(j_p_av, j0_p_av, pos)
            # pot_model = pybamm.potential.Potential(param)
            # ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n_av, ocp_p_av)
            # eta_r_vars = pot_model.get_derived_reaction_overpotentials(
            #     eta_r_n_av, eta_r_p_av
            # )
            # self.variables.update({**ocp_vars, **eta_r_vars})
            # j_vars = int_curr_model.get_derived_interfacial_currents(
            #     j_n_av, j_p_av, j0_n_av, j0_p_av
            # )
            pot_vars = eleclyte_current_model.get_first_order_potential_differences(
                self.variables, int_curr_model, self.options["first-order potential"]
            )
            self.variables.update(pot_vars)
        else:

            delta_phi_n_av = pybamm.Variable(
                "Average neg electrode surface potential difference"
            )
            delta_phi_p_av = pybamm.Variable(
                "Average pos electrode surface potential difference"
            )

            eta_r_n_av = delta_phi_n_av - ocp_n_av
            eta_r_p_av = delta_phi_p_av - ocp_p_av
            j_n_av = int_curr_model.get_butler_volmer(j0_n_av, eta_r_n_av, neg)
            j_p_av = int_curr_model.get_butler_volmer(j0_p_av, eta_r_p_av, pos)

            # Make dictionaries to pass to submodel
            self.variables.update(
                {
                    "Negative electrode surface potential difference": delta_phi_n_av,
                    "Positive electrode surface potential difference": delta_phi_p_av,
                }
            )
            reactions_av = {"main": {"neg": {"aj": j_n_av}, "pos": {"aj": j_p_av}}}

            # Call submodel using average variables and average reactions
            eleclyte_current_model = pybamm.electrolyte_current.MacInnesCapacitance(
                param, self.options["capacitance"]
            )
            eleclyte_current_model.set_leading_order_system(
                self.variables, reactions_av, neg
            )
            eleclyte_current_model.set_leading_order_system(
                self.variables, reactions_av, pos
            )
            self.update(eleclyte_current_model)

    def set_current_variables(self):
        param = self.set_of_parameters

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

        # Positive electrode potential and voltage
        electrode_vars = electrode_model.get_explicit_combined(self.variables)
        self.variables.update(electrode_vars)

        # Cut-off voltage
        voltage = self.variables["Terminal voltage"]
        self.events.append(voltage - param.voltage_low_cut)

    def set_interface_variables(self, int_curr_model):
        param = self.set_of_parameters
        c_e = self.variables["Electrolyte concentration"]
        c_e_n, _, c_e_p = c_e.orphans
        phi_e = self.variables["Electrolyte potential"]
        phi_s = self.variables["Electrode potential"]
        phi_e_n, _, phi_e_p = phi_e.orphans
        phi_s_n, _, phi_s_p = phi_s.orphans

        # Potentials
        ocp_n = param.U_n(c_e_n)
        ocp_p = param.U_p(c_e_p)
        pot_model = pybamm.potential.Potential(param)
        potential_vars = pot_model.get_all_potentials(
            (ocp_n, ocp_p), delta_phi=(phi_s_n - phi_e_n, phi_s_p - phi_e_p)
        )
        self.variables.update(potential_vars)

        # Exchange-current density
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p)
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        phi_e_0 = pybamm.average(self.leading_order_variables["Electrolyte potential"])
        phi_s_p_0 = pybamm.average(
            self.leading_order_variables["Electrode potential"].orphans[2]
        )
        delta_phi_n_0 = -phi_e_0
        delta_phi_p_0 = phi_s_p_0 - phi_e_0

        # Take 1 * c_e_0 so that it doesn't appear in delta_phi_n_0 and delta_phi_p_0
        c_e_0 = 1 * self.leading_order_variables["Average electrolyte concentration"]

        j_n_0 = int_curr_model.get_butler_volmer_from_variables(
            c_e_0, delta_phi_n_0, neg
        )
        j_p_0 = int_curr_model.get_butler_volmer_from_variables(
            c_e_0, delta_phi_p_0, pos
        )
        c_e_n_1 = (c_e_n - c_e_0) / param.C_e
        c_e_p_1 = (c_e_p - c_e_0) / param.C_e
        delta_phi_n_1 = (phi_s_n - phi_e_n - delta_phi_n_0) / param.C_e
        delta_phi_p_1 = (phi_s_p - phi_e_p - delta_phi_p_0) / param.C_e

        djn0_dce0 = self.variables["d(j_n_0)/d(c_e_0)"]
        djp0_dce0 = self.variables["d(j_p_0)/d(c_e_0)"]
        djn0_dpn0 = self.variables["d(j_n_0)/d(delta_phi_n_0)"]
        djp0_dpp0 = self.variables["d(j_p_0)/d(delta_phi_p_0)"]
        j_n_1 = djn0_dce0 * c_e_n_1 + djn0_dpn0 * delta_phi_n_1
        j_p_1 = djp0_dce0 * c_e_p_1 + djp0_dpp0 * delta_phi_p_1
        j_n = j_n_0 + param.C_e * j_n_1
        j_p = j_p_0 + param.C_e * j_p_1

        # j_n = int_curr_model.get_butler_volmer(j0_n, eta_r_n)
        # j_p = int_curr_model.get_butler_volmer(j0_p, eta_r_p)
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

    def set_convection_variables(self):
        velocity_model = pybamm.velocity.Velocity(self.set_of_parameters)
        if self.options["convection"] is not False:
            velocity_vars = velocity_model.get_explicit_composite(self.variables)
            self.variables.update(velocity_vars)
        else:
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            v_box = pybamm.Broadcast(0, whole_cell)
            dVbox_dz = pybamm.Broadcast(0, whole_cell)
            self.variables.update(velocity_model.get_variables(v_box, dVbox_dz))

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
