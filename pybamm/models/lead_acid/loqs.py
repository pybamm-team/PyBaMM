#
# Lead-acid LOQS model
#
import pybamm


class LOQS(pybamm.LeadAcidBaseModel):
    """Leading-Order Quasi-Static model for lead-acid, from [1]_.

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.

    **Extends:** :class:`pybamm.LeadAcidBaseModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "LOQS model"

        self.set_model_variables()
        self.set_boundary_conditions(self.variables)
        self.set_interface_submodel()
        self.set_electrolyte_current_submodel()
        self.set_porosity_submodel()
        self.set_diffusion_submodels()
        self.set_current_variables()
        self.set_convection_variables()

        # ODEs only (don't use jacobian)
        self.use_jacobian = False

    def set_model_variables(self):
        "Set variables for the model"
        if self.options["bc_options"]["dimensionality"] == 0:
            curr_coll_domain = []
        elif self.options["bc_options"]["dimensionality"] == 1:
            curr_coll_domain = ["current collector"]

        c_e = pybamm.Variable("Electrolyte concentration", curr_coll_domain)

        # Piecewise constant epsilon
        eps_n_pc = pybamm.Variable("Negative electrode porosity", curr_coll_domain)
        eps_s_pc = pybamm.Variable("Separator porosity", curr_coll_domain)
        eps_p_pc = pybamm.Variable("Positive electrode porosity", curr_coll_domain)

        epsilon = pybamm.Concatenation(
            pybamm.Broadcast(eps_n_pc, ["negative electrode"]),
            pybamm.Broadcast(eps_s_pc, ["separator"]),
            pybamm.Broadcast(eps_p_pc, ["positive electrode"]),
        )

        self.variables.update({"Electrolyte concentration": c_e, "Porosity": epsilon})

        if self.options["capacitance"] is not False:
            delta_phi_n = pybamm.Variable(
                "Leading-order negative electrode surface potential difference",
                curr_coll_domain,
            )
            delta_phi_p = pybamm.Variable(
                "Leading-order positive electrode surface potential difference",
                curr_coll_domain,
            )
            self.variables.update(
                {
                    "Negative electrode surface potential difference": delta_phi_n,
                    "Positive electrode surface potential difference": delta_phi_p,
                }
            )
        if "oxygen" in self.options["side reactions"]:
            c_ox = pybamm.Variable("Oxygen concentration", curr_coll_domain)
            self.variables["Oxygen concentration"] = c_ox

    def set_boundary_conditions(self, bc_variables=None):
        "Set boundary conditions, dependent on self.options"
        # TODO: edit to allow constant-current and constant-power control
        param = self.set_of_parameters
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality == 0:
            current_bc = param.current_with_time
            self.variables["Current collector current density"] = current_bc
        elif dimensionality == 1:
            current_collector_model = pybamm.vertical.Vertical(param)
            current_collector_model.set_leading_order_vertical_current(bc_variables)
            self.update(current_collector_model)

    def set_interface_submodel(self):
        if self.options["capacitance"] is False:
            self.set_interface_direct_formulation()
        else:
            self.set_interface_capacitance_formulation()
        self.set_interfacial_surface_area_submodel()
        self.set_reactions()

    def set_interface_direct_formulation(self):
        # Set up
        param = self.set_of_parameters
        c_e = self.variables["Electrolyte concentration"]
        i_boundary_cc = self.variables["Current collector current density"]
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        main_curr_model = pybamm.interface_lead_acid.MainReaction(param)
        pot_model = pybamm.potential.Potential(param)

        # Interfacial parameters
        j0_n = main_curr_model.get_exchange_current_densities(c_e, neg)
        j0_p = main_curr_model.get_exchange_current_densities(c_e, pos)
        ocp_n = param.U_n(c_e)
        ocp_p = param.U_p(c_e)

        # Interfacial current density
        j_n = main_curr_model.get_homogeneous_interfacial_current(i_boundary_cc, neg)
        j_p = main_curr_model.get_homogeneous_interfacial_current(i_boundary_cc, pos)

        # Potentials
        eta_r_n = main_curr_model.get_inverse_butler_volmer(j_n, j0_n, neg)
        eta_r_p = main_curr_model.get_inverse_butler_volmer(j_p, j0_p, pos)
        pot_vars = pot_model.get_all_potentials(
            (ocp_n, ocp_p), eta_r=(eta_r_n, eta_r_p)
        )
        self.variables.update(pot_vars)

        # Update variables
        j_vars = main_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

    def set_interface_capacitance_formulation(self):
        # Set up
        param = self.set_of_parameters
        c_e = self.variables["Electrolyte concentration"]
        delta_phi_n = self.variables["Negative electrode surface potential difference"]
        delta_phi_p = self.variables["Positive electrode surface potential difference"]
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        pot_model = pybamm.potential.Potential(param)
        self.reactions = {}

        # Main reaction
        main_curr_model = pybamm.interface_lead_acid.MainReaction(param)
        j0_n = main_curr_model.get_exchange_current_densities(c_e, neg)
        j0_p = main_curr_model.get_exchange_current_densities(c_e, pos)
        ocp_n = param.U_n(c_e)
        ocp_p = param.U_p(c_e)
        eta_r_n = delta_phi_n - ocp_n
        eta_r_p = delta_phi_p - ocp_p
        j_n = main_curr_model.get_butler_volmer(j0_n, eta_r_n, neg)
        j_p = main_curr_model.get_butler_volmer(j0_p, eta_r_p, pos)

        # Update variables
        j_vars = main_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)
        pot_vars = pot_model.get_all_potentials(
            (ocp_n, ocp_p), (eta_r_n, eta_r_p), (delta_phi_n, delta_phi_p)
        )
        self.variables.update(pot_vars)

        # Oxygen reaction
        oxygen_curr_model = pybamm.interface_lead_acid.OxygenReaction(param)
        if "oxygen" in self.options["side reactions"]:
            c_ox = self.variables["Oxygen concentration"]
            j0_p_Ox = oxygen_curr_model.get_exchange_current_densities(c_e, c_ox, pos)
            ocp_p_Ox = param.U_p_Ox
            eta_r_p_Ox = delta_phi_p - ocp_p_Ox
            j_p_Ox = oxygen_curr_model.get_tafel(j0_p_Ox, eta_r_p_Ox, pos)
            j_n_Ox = -j_p_Ox * param.l_p / param.l_n
            # Update variables
            j_Ox_vars = oxygen_curr_model.get_derived_interfacial_currents(
                j_n_Ox, j_p_Ox, pybamm.Scalar(0), j0_p_Ox
            )
        else:
            z = pybamm.Scalar(0)
            j_Ox_vars = oxygen_curr_model.get_derived_interfacial_currents(z, z, z, z)
        self.variables.update(j_Ox_vars)

    def set_interfacial_surface_area_submodel(self):
        param = self.set_of_parameters
        surface_area_model = pybamm.interface_lead_acid.InterfacialSurfaceArea(param)
        if self.options["interfacial surface area"] == "variable":
            neg = ["negative electrode"]
            pos = ["positive electrode"]
            surface_area_model.set_differential_system(self.variables, neg)
            surface_area_model.set_differential_system(self.variables, pos)
            self.update(surface_area_model)
        # else:
        #     surface_area_vars = surface_area_model.get_variables(0)
        #     self.variables.update(surface_area_vars)

    def set_reactions(self):
        param = self.set_of_parameters
        j_n = self.variables["Negative electrode interfacial current density"]
        j_p = self.variables["Positive electrode interfacial current density"]
        self.reactions = {
            "main": {
                "neg": {"s": -(param.s_plus_n_S + param.t_plus), "aj": j_n},
                "pos": {"s": -(param.s_plus_p_S + param.t_plus), "aj": j_p},
            }
        }
        if "oxygen" in self.options["side reactions"]:
            j_n_Ox = self.variables[
                "Negative electrode oxygen interfacial current density"
            ]
            j_p_Ox = self.variables[
                "Positive electrode oxygen interfacial current density"
            ]
            # Update reactions and variables
            self.reactions["oxygen"] = {
                "neg": {
                    "s": -(param.s_plus_Ox + param.t_plus),
                    "s_ox": -param.s_ox_Ox,
                    "aj": j_n_Ox,
                },
                "pos": {
                    "s": -(param.s_plus_Ox + param.t_plus),
                    "s_ox": -param.s_ox_Ox,
                    "aj": j_p_Ox,
                },
            }
            self.reactions["main"]["neg"]["s_ox"] = 0
            self.reactions["main"]["pos"]["s_ox"] = 0

    def set_electrolyte_current_submodel(self):
        if self.options["capacitance"] is not False:
            eleclyte_current_model = pybamm.electrolyte_current.MacInnesCapacitance(
                self.set_of_parameters, self.options["capacitance"]
            )
            eleclyte_current_model.set_leading_order_system(
                self.variables, self.reactions, ["negative electrode"]
            )
            eleclyte_current_model.set_leading_order_system(
                self.variables, self.reactions, ["positive electrode"]
            )
            self.update(eleclyte_current_model)

    def set_porosity_submodel(self):
        param = self.set_of_parameters
        porosity_model = pybamm.porosity.Standard(param)
        porosity_model.set_leading_order_system(self.variables)
        self.update(porosity_model)

        # Update reactions
        deps_n_dt = self.variables["Negative electrode porosity change"].orphans[0]
        deps_p_dt = self.variables["Positive electrode porosity change"].orphans[0]
        for name, reaction in self.reactions.items():
            # Only main reaction contributes to porosity change
            if name == "main":
                reaction["neg"]["deps_dt"] = deps_n_dt
                reaction["pos"]["deps_dt"] = deps_p_dt
            else:
                reaction["neg"]["deps_dt"] = 0
                reaction["pos"]["deps_dt"] = 0

    def set_diffusion_submodels(self):
        param = self.set_of_parameters
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_leading_order_system(self.variables, self.reactions)
        self.update(eleclyte_conc_model)
        oxygen_conc_model = pybamm.oxygen_diffusion.StefanMaxwell(param)
        if "oxygen" in self.options["side reactions"]:
            oxygen_conc_model.set_leading_order_system(self.variables, self.reactions)
            self.update(oxygen_conc_model)
        else:
            zero = pybamm.Scalar(0)
            oxygen_conc_vars = oxygen_conc_model.get_variables(zero, zero)
            self.variables.update(oxygen_conc_vars)

    def set_current_variables(self):
        param = self.set_of_parameters

        elyte_postproc_model = pybamm.electrolyte_current.ElectrolyteCurrentBaseModel(
            param
        )
        # Electrolyte current
        elyte_vars = elyte_postproc_model.get_explicit_leading_order(self.variables)
        self.variables.update(elyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        electrode_vars = electrode_model.get_explicit_leading_order(self.variables)
        self.variables.update(electrode_vars)

        # Cut-off voltage
        # Hack to extract voltage at the tabs in 2D
        voltage = self.variables["Terminal voltage"]
        if self.options["bc_options"]["dimensionality"] == 1:
            voltage.domain = "current collector"
            voltage = pybamm.boundary_value(voltage, "right")
        self.events.append(voltage - param.voltage_low_cut)

    def set_convection_variables(self):
        velocity_model = pybamm.velocity.Velocity(self.set_of_parameters)
        if self.options["convection"] is not False:
            velocity_vars = velocity_model.get_explicit_leading_order(self.variables)
            self.variables.update(velocity_vars)
        else:
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            v_box = pybamm.Broadcast(0, whole_cell)
            dVbox_dz = pybamm.Broadcast(0, whole_cell)
            self.variables.update(velocity_model.get_variables(v_box, dVbox_dz))

    @property
    def default_spatial_methods(self):
        # ODEs only in the macroscale, so use base spatial method
        return {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.FiniteVolume,
        }

    @property
    def default_geometry(self):
        if self.options["bc_options"]["dimensionality"] == 0:
            return pybamm.Geometry("1D macro")
        elif self.options["bc_options"]["dimensionality"] == 1:
            return pybamm.Geometry("1+1D macro")

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        if self.options["capacitance"] == "algebraic":
            return pybamm.ScikitsDaeSolver()
        else:
            return pybamm.ScipySolver()
