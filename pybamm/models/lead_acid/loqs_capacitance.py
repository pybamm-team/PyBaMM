#
# Lead-acid LOQS model, with capacitance
#
import pybamm


class LOQSCapacitance(pybamm.LeadAcidBaseModel):
    """
    Leading-Order Quasi-Static model for lead-acid, from [2]_, with capacitance
    effects included

    Parameters
    ----------
    use_capacitance : bool
        Whether to use capacitance in the model or not. If True (default), solve
        ODEs for delta_phi. If False, solve algebraic equations for delta_phi

    References
    ----------
    .. [2] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.

    **Extends**: :class:`pybamm.LeadAcidBaseModel`
    """

    def __init__(self, use_capacitance=True, bc_options=None):
        super().__init__()
        self._use_capacitance = use_capacitance
        self.bc_options = bc_options or self.default_bc_options

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid
        self.param = param

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.Variable("Electrolyte concentration")
        delta_phi_n = pybamm.Variable("Negative electrode potential difference")
        delta_phi_p = pybamm.Variable("Positive electrode potential difference")
        epsilon = pybamm.standard_variables.eps_piecewise_constant

        "-----------------------------------------------------------------------------"
        "Boundary conditions"
        bc_variables = {"delta_phi_n": delta_phi_n, "delta_phi_p": delta_phi_p}
        self.set_boundary_conditions(bc_variables)

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
        eleclyte_current_model_n = pybamm.electrolyte_current.MacInnesCapacitance(
            param, use_capacitance
        )
        eleclyte_current_model_n.set_leading_order_system(delta_phi_n, reactions, neg)
        eleclyte_current_model_p = pybamm.electrolyte_current.MacInnesCapacitance(
            param, use_capacitance
        )
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
        electrolyte_vars = eleclyte_current_model_p.get_explicit_leading_order(
            ocp_n, eta_r_n
        )
        self.variables.update(electrolyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        phi_e = self.variables["Electrolyte potential"]
        electrode_vars = electrode_model.get_explicit_leading_order(
            ocp_p, eta_r_p, phi_e
        )
        self.variables.update(electrode_vars)

        # Add cut-off voltage, using potential differences for quicker evaluation
        voltage = delta_phi_p - delta_phi_n
        self.events.append(voltage - param.voltage_low_cut)

        "-----------------------------------------------------------------------------"
        "Extra settings"

        # ODE model, don't use Jacobian
        self.use_jacobian = False

        # Different solver depending on whether we solve ODEs or DAEs
        if use_capacitance:
            self.default_solver = pybamm.ScikitsOdeSolver()
        else:
            self.default_solver = pybamm.ScikitsDaeSolver()
        if self.bc_options["dimensionality"] == 1:
            self.default_geometry = pybamm.Geometry("1D macro")
        elif self.bc_options["dimensionality"] == 2:
            self.default_geometry = pybamm.Geometry("1+1D macro")

    def set_boundary_conditions(self, bc_variables=None):
        """Get boundary conditions"""
        # TODO: edit to allow constant-current and constant-power control
        dimensionality = self.bc_options["dimensionality"]
        if dimensionality == 1:
            current_bc = self.param.current_with_time
            self.variables.update({"Current collector current": current_bc})
        elif dimensionality == 2:
            delta_phi_n = bc_variables["delta_phi_n"]
            delta_phi_p = bc_variables["delta_phi_p"]
            current_collector_model = pybamm.vertical.Vertical(self.param)
            current_collector_model.set_leading_order_vertical_current(
                delta_phi_n, delta_phi_p
            )
            self.update(current_collector_model)
