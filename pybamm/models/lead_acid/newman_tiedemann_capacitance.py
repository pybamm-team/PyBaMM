#
# Lead-acid Newman-Tiedemann model, including capacitance effects
#
import pybamm


class NewmanTiedemannCapacitance(pybamm.LeadAcidBaseModel):
    """
    Porous electrode model for lead-acid, from [2]_, with capacitance effects
    included.

    Parameters
    ----------
    use_capacitance : bool
        Whether to use capacitance in the model or not. If True (default), solve
        ODEs for delta_phi. If False, solve algebraic equations for delta_phi

    References
    ----------
    .. [2] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: I. Physical Model.
           arXiv preprint arXiv:1902.01771, 2019.

    **Extends:** :class:`pybamm.LeadAcidBaseModel`
    """

    def __init__(self, use_capacitance=True):
        super().__init__()
        self._use_capacitance = use_capacitance

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.standard_variables.c_e
        eps = pybamm.standard_variables.eps
        delta_phi_n = pybamm.standard_variables.delta_phi_n
        delta_phi_p = pybamm.standard_variables.delta_phi_p

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Exchange-current density
        c_e_n, c_e_s, c_e_p = c_e.orphans
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

        # Electrolyte current
        eps_n, _, eps_p = eps.orphans
        eleclyte_current_model_n = pybamm.electrolyte_current.MacInnesCapacitance(
            param, use_capacitance
        )
        eleclyte_current_model_n.set_full_system(delta_phi_n, c_e_n, reactions, eps_n)
        eleclyte_current_model_p = pybamm.electrolyte_current.MacInnesCapacitance(
            param, use_capacitance
        )
        eleclyte_current_model_p.set_full_system(delta_phi_p, c_e_p, reactions, eps_p)

        "-----------------------------------------------------------------------------"
        "Combine Submodels"
        self.update(
            porosity_model,
            electrolyte_diffusion_model,
            eleclyte_current_model_n,
            eleclyte_current_model_p,
        )

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

        # Post-process electrolyte model
        i_e_n = self.variables["Negative electrolyte current density"]
        i_e_p = self.variables["Positive electrolyte current density"]
        eleclyte_variables = eleclyte_current_model_p.get_post_processed(
            delta_phi_n, delta_phi_p, i_e_n, i_e_p, c_e, eps
        )
        self.variables.update(eleclyte_variables)
        # self.boundary_conditions.update(
        #     {
        #         c_e_n: {
        #             "left": (pybamm.Scalar(0), "Neumann"),
        #             "right": (pybamm.BoundaryFlux(c_e_n, "right"), "Neumann"),
        #         },
        #         c_e_s: {
        #             "left": (pybamm.BoundaryFlux(c_e_s, "left"), "Neumann"),
        #             "right": (pybamm.BoundaryFlux(c_e_s, "right"), "Neumann"),
        #         },
        #         c_e_p: {
        #             "left": (pybamm.BoundaryFlux(c_e_p, "left"), "Neumann"),
        #             "right": (pybamm.Scalar(0), "Neumann"),
        #         },
        #     }
        # )
        # Voltage
        phi_e = self.variables["Electrolyte potential"]
        phi_e_n, _, phi_e_p = phi_e.orphans
        phi_s_n = delta_phi_n + phi_e_n
        phi_s_p = delta_phi_p + phi_e_p
        i_cell = param.current_with_time
        i_s_n = i_cell - i_e_n
        i_s_p = i_cell - i_e_p
        electrode_current_model = pybamm.electrode.Ohm(param)
        vol_vars = electrode_current_model.get_variables(phi_s_n, phi_s_p, i_s_n, i_s_p)
        self.variables.update(vol_vars)

        # Rough voltage cut-off
        voltage = pybamm.BoundaryValue(delta_phi_p, "right") - pybamm.BoundaryValue(
            delta_phi_n, "left"
        )
        self.events.append(voltage - param.voltage_low_cut)

        "-----------------------------------------------------------------------------"
        "Extra settings"

        # Different solver depending on whether we solve ODEs or DAEs
        if use_capacitance:
            self.default_solver = pybamm.ScikitsOdeSolver()
        else:
            self.default_solver = pybamm.ScikitsDaeSolver()
