#
# Single Particle Model with Electrolyte (SPMe)
#
import pybamm


class SPMe(pybamm.LithiumIonBaseModel):
    """Single Particle Model with Electrolyte (SPMe) of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self):
        super().__init__()
        self.name = "Single Particle Model with electrolyte"

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lithium_ion
        i_boundary_cc = param.current_with_time
        self.variables["Current collector current density"] = i_boundary_cc

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_s_n = pybamm.standard_variables.c_s_n
        c_s_p = pybamm.standard_variables.c_s_p
        c_e = pybamm.standard_variables.c_e
        self.variables["Electrolyte concentration"] = c_e

        "-----------------------------------------------------------------------------"
        "Submodels"
        # Interfacial current density
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        int_curr_model = pybamm.interface.LithiumIonReaction(param)
        j_n = int_curr_model.get_homogeneous_interfacial_current(i_boundary_cc, neg)
        j_p = int_curr_model.get_homogeneous_interfacial_current(i_boundary_cc, pos)

        # Particle models
        negative_particle_model = pybamm.particle.Standard(param)
        negative_particle_model.set_differential_system(c_s_n, j_n, broadcast=True)
        positive_particle_model = pybamm.particle.Standard(param)
        positive_particle_model.set_differential_system(c_s_p, j_p, broadcast=True)

        # Electrolyte concentration
        broad_j_n = pybamm.Broadcast(j_n, ["negative electrode"])
        broad_j_p = pybamm.Broadcast(j_p, ["positive electrode"])
        reactions = {
            "main": {"neg": {"s": 1, "aj": broad_j_n}, "pos": {"s": 1, "aj": broad_j_p}}
        }
        # Electrolyte diffusion model
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        electrolyte_diffusion_model.set_differential_system(self.variables, reactions)

        self.update(
            negative_particle_model,
            positive_particle_model,
            electrolyte_diffusion_model,
        )

        "-----------------------------------------------------------------------------"
        "Post-Processing"
        # Exchange-current density
        c_e_n, _, c_e_p = c_e.orphans
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n, c_s_n_surf, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p, c_s_p_surf, pos)
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

        # OCP and Overpotentials
        ocp_n = param.U_n(c_s_n_surf)
        ocp_p = param.U_p(c_s_p_surf)
        eta_r_n = int_curr_model.get_inverse_butler_volmer(j_n, j0_n, neg)
        eta_r_p = int_curr_model.get_inverse_butler_volmer(j_p, j0_p, pos)
        pot_model = pybamm.potential.Potential(param)
        pot_vars = pot_model.get_all_potentials(
            (ocp_n, ocp_p), eta_r=(eta_r_n, eta_r_p)
        )
        self.variables.update(pot_vars)

        # Load electrode and electrolyte models
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

        # Positive electrode potential
        electrode_vars = electrode_model.get_explicit_combined(self.variables)
        self.variables.update(electrode_vars)

        # Update potentials again to get correct surface potential difference
        delta_phi_n = (
            self.variables["Negative electrode potential"]
            - self.variables["Negative electrolyte potential"]
        )
        delta_phi_p = (
            self.variables["Positive electrode potential"]
            - self.variables["Positive electrolyte potential"]
        )
        pot_vars = pot_model.get_all_potentials(
            (ocp_n, ocp_p), delta_phi=(delta_phi_n, delta_phi_p)
        )
        self.variables.update(pot_vars)

        # Cut-off voltage
        voltage = self.variables["Terminal voltage"]
        self.events["Minimum voltage cut-off"] = voltage - param.voltage_low_cut

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro", "1D micro")
