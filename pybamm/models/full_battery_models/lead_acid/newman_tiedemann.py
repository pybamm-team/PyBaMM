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

        self.set_model_variables()
        self.set_boundary_conditions(None)
        self.set_interfacial_variables()
        self.set_porosity_submodel()
        self.set_convection_submodel()
        self.set_diffusion_submodel()
        self.set_current_submodels()

    def set_model_variables(self):
        c_e = pybamm.standard_variables.c_e
        eps = pybamm.standard_variables.eps

        if self.options["capacitance"] is False:
            phi_e = pybamm.standard_variables.phi_e
            phi_s_p = pybamm.standard_variables.phi_s_p
            phi_s_n = pybamm.standard_variables.phi_s_n
            delta_phi_n = phi_s_n - phi_e.orphans[0]
            delta_phi_p = phi_s_p - phi_e.orphans[2]
            self.variables.update(
                {
                    "Electrolyte potential": phi_e,
                    "Negative electrode potential": phi_s_n,
                    "Positive electrode potential": phi_s_p,
                }
            )
        else:
            delta_phi_n = pybamm.standard_variables.delta_phi_n
            delta_phi_p = pybamm.standard_variables.delta_phi_p

        self.variables.update(
            {
                "Electrolyte concentration": c_e,
                "Porosity": eps,
                "Negative electrode surface potential difference": delta_phi_n,
                "Positive electrode surface potential difference": delta_phi_p,
            }
        )

    def set_boundary_conditions(self, bc_variables):
        """Set boundary conditions, dependent on self.options"""
        param = self.set_of_parameters
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality == 0:
            current_bc = param.current_with_time
            self.variables["Current collector current density"] = current_bc

    def set_interfacial_variables(self):
        param = self.set_of_parameters

        # Exchange-current density
        int_curr_model = pybamm.interface.LeadAcidReaction(param)
        c_e = self.variables["Electrolyte concentration"]
        c_e_n, _, c_e_p = c_e.orphans

        # Potentials
        delta_phi_n = self.variables["Negative electrode surface potential difference"]
        delta_phi_p = self.variables["Positive electrode surface potential difference"]
        ocp_n = param.U_n(c_e_n)
        ocp_p = param.U_p(c_e_p)
        eta_r_n = delta_phi_n - ocp_n
        eta_r_p = delta_phi_p - ocp_p
        pot_model = pybamm.potential.Potential(param)
        ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n, ocp_p)
        eta_r_vars = pot_model.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
        self.variables.update({**ocp_vars, **eta_r_vars})

        # Interfacial current density
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p)
        j_n = int_curr_model.get_butler_volmer(j0_n, eta_r_n)
        j_p = int_curr_model.get_butler_volmer(j0_p, eta_r_p)
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

        # Reactions
        self.reactions = {
            "main": {
                "neg": {"s_plus": param.s_n, "aj": j_n},
                "pos": {"s_plus": param.s_p, "aj": j_p},
            }
        }

        # Cut-off voltage
        voltage = pybamm.BoundaryValue(delta_phi_p, "right") - pybamm.BoundaryValue(
            delta_phi_n, "left"
        )
        self.events.append(voltage - param.voltage_low_cut)

    def set_porosity_submodel(self):
        porosity_model = pybamm.porosity.Standard(self.set_of_parameters)
        porosity_model.set_differential_system(self.variables)
        self.update(porosity_model)

        # Update reactions
        self.reactions["main"]["porosity change"] = porosity_model.variables[
            "Porosity change"
        ]

    def set_convection_submodel(self):
        velocity_model = pybamm.velocity.Velocity(self.set_of_parameters)
        if self.options["convection"] is not False:
            self.variables["Electrolyte pressure"] = pybamm.standard_variables.pressure
            velocity_model.set_algebraic_system(self.variables)
            self.update(velocity_model)
        else:
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            v_box = pybamm.Broadcast(0, whole_cell)
            dVbox_dz = pybamm.Broadcast(0, whole_cell)
            self.variables.update(velocity_model.get_variables(v_box, dVbox_dz))

    def set_diffusion_submodel(self):
        param = self.set_of_parameters
        reactions = self.reactions

        # Electrolyte diffusion model
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        electrolyte_diffusion_model.set_differential_system(self.variables, reactions)
        self.update(electrolyte_diffusion_model)

    def set_current_submodels(self):
        param = self.set_of_parameters
        variables = self.variables
        reactions = self.reactions
        neg = ["negative electrode"]
        pos = ["positive electrode"]

        if self.options["capacitance"] is False:
            eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
                param
            )
            eleclyte_current_model.set_algebraic_system(self.variables, reactions)

            # Electrode model
            electrode_current_model = pybamm.electrode.Ohm(param)
            electrode_current_model.set_algebraic_system(variables, reactions, neg)
            electrode_current_model.set_algebraic_system(variables, reactions, pos)
            self.update(eleclyte_current_model, electrode_current_model)
            phi_s_n = variables["Negative electrode potential"]
            phi_s_p = variables["Positive electrode potential"]
            i_s_n = variables["Negative electrode current density"]
            i_s_p = variables["Positive electrode current density"]
            volt_vars = electrode_current_model.get_variables(
                phi_s_n, phi_s_p, i_s_n, i_s_p
            )
            self.variables.update(volt_vars)
        else:
            # Electrolyte current
            eleclyte_current_model = pybamm.electrolyte_current.MacInnesCapacitance(
                param, self.options["capacitance"]
            )
            eleclyte_current_model.set_full_system(variables, reactions, neg)
            eleclyte_current_model.set_full_system(variables, reactions, pos)

            # Post-process electrolyte model
            eleclyte_current_model.set_post_processed()
            self.update(eleclyte_current_model)

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
