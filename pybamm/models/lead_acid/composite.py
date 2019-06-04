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
    - Solve for electrolyte concentration, using leading-order porosity and uniform \
    interfacial current density
    - Calculate average first-order surface potential differences
    - Calculate first-order electrolyte and electrode potentials, using average \
    surface potential differences
    - Calcualte first-order surface potential-differences and interfacial current \
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
        int_curr_model = pybamm.interface_lead_acid.MainReaction(self.set_of_parameters)
        self.set_diffusion_submodel()
        self.set_electrolyte_current_model(int_curr_model)
        self.set_current_variables()
        self.set_interface_variables(int_curr_model)
        self.set_convection_variables()

    def set_boundary_conditions(self, bc_variables):
        """Set boundary conditions, dependent on self.options"""
        pybamm.logger.debug("Creating boundary-conditions submodel")
        param = self.set_of_parameters
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality == 0:
            current_bc = param.current_with_time
            self.variables["Current collector current density"] = current_bc

    def set_diffusion_submodel(self):
        pybamm.logger.debug("Creating diffusion submodel")
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
        pybamm.logger.debug("Creating electrolyte current submodel")
        if self.options["first-order potential"] == "linear":
            pot_vars = int_curr_model.get_first_order_potential_differences(
                self.variables, self.leading_order_variables
            )
        elif self.options["first-order potential"] == "average":
            pot_vars = int_curr_model.get_average_potential_differences(self.variables)
        self.variables.update(pot_vars)

    def set_current_variables(self):
        pybamm.logger.debug("Setting current variables")
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
            self.variables, first_order="linear"
        )
        self.variables.update(electrolyte_vars)

        # Positive electrode potential and voltage
        electrode_vars = electrode_model.get_explicit_combined(self.variables)
        self.variables.update(electrode_vars)

    def set_interface_variables(self, int_curr_model):
        pybamm.logger.debug("Setting interface variables")
        param = self.set_of_parameters
        c_e = self.variables["Electrolyte concentration"]
        c_e_n, _, c_e_p = c_e.orphans
        phi_e = self.variables["Electrolyte potential"]
        phi_s = self.variables["Electrode potential"]
        phi_e_n, _, phi_e_p = phi_e.orphans
        phi_s_n, _, phi_s_p = phi_s.orphans
        delta_phi_n = phi_s_n - phi_e_n
        delta_phi_p = phi_s_p - phi_e_p

        # Potentials
        ocp_n = param.U_n(c_e_n)
        ocp_p = param.U_p(c_e_p)
        pot_model = pybamm.potential.Potential(param)
        potential_vars = pot_model.get_all_potentials(
            (ocp_n, ocp_p), delta_phi=(delta_phi_n, delta_phi_p)
        )
        self.variables.update()

        # Exchange-current density
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p)
        if self.options["first-order potential"] == "linear":
            delta_phi_n_0 = pybamm.average(
                self.leading_order_variables[
                    "Negative electrode surface potential difference"
                ]
            )
            delta_phi_p_0 = pybamm.average(
                self.leading_order_variables[
                    "Positive electrode surface potential difference"
                ]
            )
            c_e_0 = self.leading_order_variables["Average electrolyte concentration"]

            j_n = int_curr_model.get_first_order_butler_volmer(
                c_e_n, delta_phi_n, c_e_0, delta_phi_n_0
            )
            j_p = int_curr_model.get_first_order_butler_volmer(
                c_e_p, delta_phi_p, c_e_0, delta_phi_p_0
            )
        elif self.options["first-order potential"] == "average":
            j_n = int_curr_model.get_butler_volmer_from_variables(c_e_n, delta_phi_n)
            j_p = int_curr_model.get_butler_volmer_from_variables(c_e_p, delta_phi_p)
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update({**potential_vars, **j_vars})

        # Update current
        pybamm.logger.debug("Updating current variables with interfacial current")
        current_variables = int_curr_model.get_current_from_current_densities(
            self.variables
        )
        self.variables.update(current_variables)

    def set_convection_variables(self):
        pybamm.logger.debug("Setting convection variables")
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
