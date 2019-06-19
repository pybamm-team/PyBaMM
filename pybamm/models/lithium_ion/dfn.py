#
# Doyle-Fuller-Newman (DFN) Model
#
import pybamm


class DFN(pybamm.LithiumIonBaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self):
        super().__init__()
        self.name = "Doyle-Fuller-Newman model"

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lithium_ion
        i_boundary_cc = param.current_with_time
        self.variables["Current collector current density"] = i_boundary_cc

        "-----------------------------------------------------------------------------"
        "Model Variables"

        self.set_model_variables()
        c_e = self.variables["Electrolyte concentration"]
        c_s_n = self.variables["Negative particle concentration"]
        c_s_p = self.variables["Positive particle concentration"]
        delta_phi_n = self.variables["Negative electrode surface potential difference"]
        delta_phi_p = self.variables["Positive electrode surface potential difference"]

        "-----------------------------------------------------------------------------"
        "Submodels"
        # Exchange-current density
        c_e_n, _, c_e_p = c_e.orphans
        c_s_n_surf = pybamm.surf(c_s_n, set_domain=True)
        c_s_p_surf = pybamm.surf(c_s_p, set_domain=True)
        int_curr_model = pybamm.interface.LithiumIonReaction(param)
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n, c_s_n_surf)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p, c_s_p_surf)

        # Potentials
        ocp_n = param.U_n(c_s_n_surf)
        ocp_p = param.U_p(c_s_p_surf)
        eta_r_n = delta_phi_n - ocp_n
        eta_r_p = delta_phi_p - ocp_p

        # Interfacial current density
        j_n = int_curr_model.get_butler_volmer(j0_n, eta_r_n)
        j_p = int_curr_model.get_butler_volmer(j0_p, eta_r_p)

        # Particle models
        negative_particle_model = pybamm.particle.Standard(param)
        negative_particle_model.set_differential_system(c_s_n, j_n)
        positive_particle_model = pybamm.particle.Standard(param)
        positive_particle_model.set_differential_system(c_s_p, j_p)

        # Electrolyte concentration
        reactions = {"main": {"neg": {"s": 1, "aj": j_n}, "pos": {"s": 1, "aj": j_p}}}
        # Electrolyte diffusion model
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        electrolyte_diffusion_model.set_differential_system(self.variables, reactions)

        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        eleclyte_current_model.set_algebraic_system(self.variables, reactions)

        # Electrode models
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        electrode_current_model = pybamm.electrode.Ohm(param)
        electrode_current_model.set_algebraic_system(self.variables, reactions, neg)
        electrode_current_model.set_algebraic_system(self.variables, reactions, pos)

        "-----------------------------------------------------------------------------"
        "Combine Submodels"
        self.update(
            negative_particle_model,
            positive_particle_model,
            electrolyte_diffusion_model,
            eleclyte_current_model,
            electrode_current_model,
        )

        "-----------------------------------------------------------------------------"
        "Post-process"

        # Excahnge-current density
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

        # Potentials
        pot_model = pybamm.potential.Potential(param)
        pot_vars = pot_model.get_all_potentials(
            (ocp_n, ocp_p), eta_r=(eta_r_n, eta_r_p)
        )
        self.variables.update(pot_vars)

        # Voltage
        phi_s_n = self.variables["Negative electrode potential"]
        phi_s_p = self.variables["Positive electrode potential"]
        i_s_n = self.variables["Negative electrode current density"]
        i_s_p = self.variables["Positive electrode current density"]
        pot_vars = electrode_current_model.get_potential_variables(phi_s_n, phi_s_p)
        curr_vars = electrode_current_model.get_current_variables(i_s_n, i_s_p)
        self.variables.update({**pot_vars, **curr_vars})

        # Cut-off voltage
        voltage = self.variables["Terminal voltage"]
        self.events["Minimum voltage cut-off"] = voltage - param.voltage_low_cut

    def set_model_variables(self):
        c_s_n = pybamm.standard_variables.c_s_n
        c_s_p = pybamm.standard_variables.c_s_p
        c_e = pybamm.standard_variables.c_e
        phi_e = pybamm.standard_variables.phi_e
        phi_s_p = pybamm.standard_variables.phi_s_p
        phi_s_n = pybamm.standard_variables.phi_s_n
        delta_phi_n = phi_s_n - phi_e.orphans[0]
        delta_phi_p = phi_s_p - phi_e.orphans[2]
        self.variables.update(
            {
                "Electrolyte concentration": c_e,
                "Negative particle concentration": c_s_n,
                "Positive particle concentration": c_s_p,
                "Electrolyte potential": phi_e,
                "Negative electrode potential": phi_s_n,
                "Positive electrode potential": phi_s_p,
                "Negative electrode surface potential difference": delta_phi_n,
                "Positive electrode surface potential difference": delta_phi_p,
            }
        )

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro", "1+1D micro")

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """

        # Default solver to DAE
        return pybamm.ScikitsDaeSolver()
