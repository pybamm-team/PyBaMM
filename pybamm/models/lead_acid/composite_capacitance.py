#
# Lead-acid Composite model
#
import pybamm


class CompositeCapacitance(pybamm.LeadAcidBaseModel):
    """
    Composite model for lead-acid, from [2]_, with capacitance effects included.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

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

    **Extends:** :class:`pybamm.LeadAcidBaseModel`
    """

    def __init__(self, use_capacitance=True):
        # Update own model with submodels
        super().__init__()
        self.use_capacitance = use_capacitance

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.standard_variables.c_e
        eps = pybamm.standard_variables.eps
        delta_phi_n_av = pybamm.Variable("Average neg electrode potential difference")
        delta_phi_p_av = pybamm.Variable("Average pos electrode potential difference")

        "-----------------------------------------------------------------------------"
        "Submodels"
        # Leading order model
        leading_order_model = pybamm.lead_acid.LOQS()

        # Interfacial current density
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        int_curr_model = pybamm.interface.LeadAcidReaction(param)
        j_0_n = int_curr_model.get_homogeneous_interfacial_current(neg)
        j_0_p = int_curr_model.get_homogeneous_interfacial_current(pos)
        broad_j_0_n = pybamm.Broadcast(j_0_n, neg)
        broad_j_0_p = pybamm.Broadcast(j_0_p, pos)

        # Average composite interfacial current density
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

        # Porosity
        porosity_model = pybamm.porosity.Standard(param)
        porosity_model.set_differential_system(eps, broad_j_0_n, broad_j_0_p)

        # Electrolyte concentration
        reactions = {
            "main": {
                "neg": {"s_plus": param.s_n, "aj": broad_j_0_n},
                "pos": {"s_plus": param.s_p, "aj": broad_j_0_p},
                "porosity change": porosity_model.variables["Porosity change"],
            }
        }
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_differential_system(c_e, reactions, epsilon=eps)

        # Electrolyte current
        reactions_av = {"main": {"neg": {"aj": j_n_av}, "pos": {"aj": j_p_av}}}
        eleclyte_current_model_n = pybamm.electrolyte_current.MacInnesCapacitance(
            param, use_capacitance
        )
        eleclyte_current_model_n.set_leading_order_system(
            delta_phi_n_av, reactions_av, neg
        )
        eleclyte_current_model_p = pybamm.electrolyte_current.MacInnesCapacitance(
            param, use_capacitance
        )
        eleclyte_current_model_p.set_leading_order_system(
            delta_phi_p_av, reactions_av, pos
        )

        self.update(
            leading_order_model,
            porosity_model,
            eleclyte_conc_model,
            eleclyte_current_model_n,
            eleclyte_current_model_p,
        )

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Exchange-current density
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p, pos)
        j_vars = int_curr_model.get_derived_interfacial_currents(
            j_0_n, j_0_p, j0_n, j0_p
        )
        self.variables.update(j_vars)

        # Potentials
        ocp_n = param.U_n(c_e_n)
        ocp_p = param.U_p(c_e_p)
        eta_r_n = int_curr_model.get_inverse_butler_volmer(j_0_n, j0_n, neg)
        eta_r_p = int_curr_model.get_inverse_butler_volmer(j_0_p, j0_p, pos)
        pot_model = pybamm.potential.Potential(param)
        ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n, ocp_p)
        eta_r_vars = pot_model.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
        self.variables.update({**ocp_vars, **eta_r_vars})

        eps0 = leading_order_model.variables["Porosity"]
        c_e_0 = (
            leading_order_model.variables["Electrolyte concentration"]
            .orphans[0]
            .orphans[0]
        )

        # Load electrolyte and electrode potentials
        electrode_model = pybamm.electrode.Ohm(param)
        electrolyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
            param
        )

        # Negative electrode potential
        phi_s_n = electrode_model.get_neg_pot_explicit_combined(eps0)

        # Electrolyte potential
        electrolyte_vars = electrolyte_current_model.get_explicit_combined(
            ocp_n, eta_r_n, c_e, phi_s_n, eps0, c_e_0
        )
        self.variables.update(electrolyte_vars)
        phi_e = electrolyte_vars["Electrolyte potential"]

        # Electrode
        electrode_vars = electrode_model.get_explicit_combined(
            phi_s_n, phi_e, ocp_p, eta_r_p, eps0
        )
        self.variables.update(electrode_vars)

        "-----------------------------------------------------------------------------"
        "Extra settings"

        # Don't use jacobian for now (simplifications failing)
        self.use_jacobian = False

        # Different solver depending on whether we solve ODEs or DAEs
        if use_capacitance:
            self.default_solver = pybamm.ScikitsOdeSolver()
        else:
            self.default_solver = pybamm.ScikitsDaeSolver()
