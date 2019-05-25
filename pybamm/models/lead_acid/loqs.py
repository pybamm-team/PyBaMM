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

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid
        self._set_of_parameters = param
        capacitance_options = self.options["capacitance"]

        "-----------------------------------------------------------------------------"
        "Model Variables"

        if self.options["bc_options"]["dimensionality"] == 1:
            domain = []
        elif self.options["bc_options"]["dimensionality"] == 2:
            domain = ["current collector"]

        c_e = pybamm.Variable("Electrolyte concentration", domain)

        # Piecewise constant epsilon
        eps_n_pc = pybamm.Variable("Negative electrode porosity", domain)
        eps_s_pc = pybamm.Variable("Separator porosity", domain)
        eps_p_pc = pybamm.Variable("Positive electrode porosity", domain)

        epsilon = pybamm.Concatenation(
            pybamm.Broadcast(eps_n_pc, ["negative electrode"]),
            pybamm.Broadcast(eps_s_pc, ["separator"]),
            pybamm.Broadcast(eps_p_pc, ["positive electrode"]),
        )

        "-----------------------------------------------------------------------------"
        "Model for current"

        # Exchange-current density
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        int_curr_model = pybamm.interface.LeadAcidReaction(param)
        j0_n = int_curr_model.get_exchange_current_densities(c_e, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e, pos)

        if capacitance_options:
            delta_phi_n = pybamm.Variable(
                "Negative electrode surface potential difference", domain
            )
            delta_phi_p = pybamm.Variable(
                "Positive electrode surface potential difference", domain
            )

            bc_variables = {"delta_phi_n": delta_phi_n, "delta_phi_p": delta_phi_p}
            self.set_boundary_conditions(bc_variables)
            i_current_collector = self.variables["Current collector current"]

            # Open-circuit potential and reaction overpotential
            ocp_n = param.U_n(c_e)
            ocp_p = param.U_p(c_e)
            eta_r_n = delta_phi_n - ocp_n
            eta_r_p = delta_phi_p - ocp_p

            # Interfacial current density
            j_n = int_curr_model.get_butler_volmer(
                j0_n, eta_r_n, ["negative electrode"]
            )
            j_p = int_curr_model.get_butler_volmer(
                j0_p, eta_r_p, ["positive electrode"]
            )

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

            # Electrolyte current
            eleclyte_current_model = pybamm.electrolyte_current.MacInnesCapacitance(
                param, capacitance_options
            )
            eleclyte_current_model.set_leading_order_system(
                delta_phi_n, reactions, neg, i_current_collector
            )
            eleclyte_current_model.set_leading_order_system(
                delta_phi_p, reactions, pos, i_current_collector
            )
            self.update(eleclyte_current_model)

        else:
            i_current_collector = param.current_with_time

            # Interfacial current density
            j_n = int_curr_model.get_homogeneous_interfacial_current(
                ["negative electrode"]
            )
            j_p = int_curr_model.get_homogeneous_interfacial_current(
                ["positive electrode"]
            )

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

            # Potentials
            ocp_n = param.U_n(c_e)
            ocp_p = param.U_p(c_e)
            eta_r_n = int_curr_model.get_inverse_butler_volmer(j_n, j0_n, neg)
            eta_r_p = int_curr_model.get_inverse_butler_volmer(j_p, j0_p, pos)

            eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
                param
            )

        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_leading_order_system(c_e, reactions, epsilon=epsilon)

        self.update(porosity_model, eleclyte_conc_model)

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

        # Electrolyte current
        elyte_vars = eleclyte_current_model.get_explicit_leading_order(
            ocp_n, eta_r_n, i_current_collector
        )
        self.variables.update(elyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        phi_e = self.variables["Electrolyte potential"]
        electrode_vars = electrode_model.get_explicit_leading_order(
            ocp_p, eta_r_p, phi_e, i_current_collector
        )
        self.variables.update(electrode_vars)

        # Cut-off voltage
        voltage = self.variables["Terminal voltage"]
        self.events.append(voltage - param.voltage_low_cut)

        "-----------------------------------------------------------------------------"
        "Settings"
        # ODEs only (don't use jacobian, use base spatial method)
        self.use_jacobian = False

    @property
    def default_spatial_methods(self):
        # ODEs only in the macroscale, so use base spatial method
        return {
            "macroscale": pybamm.SpatialMethod,
            "current collector": pybamm.FiniteVolume,
        }

    @property
    def default_geometry(self):
        if self.options["bc_options"]["dimensionality"] == 1:
            return pybamm.Geometry("1D macro")
        elif self.options["bc_options"]["dimensionality"] == 2:
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
            return pybamm.ScikitsOdeSolver()

    def set_boundary_conditions(self, bc_variables=None):
        """Get boundary conditions"""
        # TODO: edit to allow constant-current and constant-power control
        param = self.set_of_parameters
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality == 1:
            current_bc = param.current_with_time
            self.variables.update({"Current collector current": current_bc})
        elif dimensionality == 2:
            delta_phi_n = bc_variables["delta_phi_n"]
            delta_phi_p = bc_variables["delta_phi_p"]
            current_collector_model = pybamm.vertical.Vertical(param)
            current_collector_model.set_leading_order_vertical_current(
                delta_phi_n, delta_phi_p
            )
            self.update(current_collector_model)
