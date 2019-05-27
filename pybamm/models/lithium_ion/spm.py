#
# Single Particle Model (SPM)
#
import pybamm


class SPM(pybamm.LithiumIonBaseModel):
    """Single Particle Model (SPM) of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self):
        super().__init__(bc_options=None)
        self.name = "Single Particle Model"
        self._bc_options = bc_options or self.default_bc_options

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lithium_ion

        "-----------------------------------------------------------------------------"
        "Model Variables"
        c_s_n = pybamm.standard_variables.c_s_n
        c_s_p = pybamm.standard_variables.c_s_p

       "-----------------------------------------------------------------------------"
        "Boundary conditions"
        # TO DO: edit below for SPM
        bc_variables = {"delta_phi_n": delta_phi_n, "delta_phi_p": delta_phi_p}
        self.set_boundary_conditions(bc_variables)
        i_current_collector = self.variables["Current collector current"]

        #if self.bc_options["dimensionality"] == 1:
        #    i_local = param.current_with_time
        #elif self.bc_options["dimensionality"] == 2 or 3:
        #    v_local = pybamm.Variable("Local cell voltage", domain="current collector")
        #    i_local = pybamm.Variable(
        #        "Local through-cell current density", domain="current collector"
        #    )


        "-----------------------------------------------------------------------------"
        "Submodels"

        # Current collector model
        if self.bc_options["dimensionality"] == 3:
            raise NotImplementedError
        if self.bc_options["dimensionality"] == 3:
            current_collector_model = pybamm.current_collector.OhmTwoDimensional(param)
            set_algebraic_system(self, v_local, i_local)

        # Interfacial current density
        int_curr_model = pybamm.interface.LithiumIonReaction(param)
        j_n = int_curr_model.get_homogeneous_interfacial_current(["negative electrode"])
        j_p = int_curr_model.get_homogeneous_interfacial_current(["positive electrode"])

        # Particle models
        negative_particle_model = pybamm.particle.Standard(param)
        negative_particle_model.set_differential_system(c_s_n, j_n, broadcast=True)
        positive_particle_model = pybamm.particle.Standard(param)
        positive_particle_model.set_differential_system(c_s_p, j_p, broadcast=True)
        self.update(negative_particle_model, positive_particle_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Electrolyte concentration
        c_e = pybamm.Scalar(1)
        N_e = pybamm.Scalar(0)
        electrolyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        conc_vars = electrolyte_conc_model.get_variables(c_e, N_e)
        self.variables.update(conc_vars)

        # Exchange-current density
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)
        j0_n = int_curr_model.get_exchange_current_densities(c_e, c_s_n_surf, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e, c_s_p_surf, pos)
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

        # Potentials
        ocp_n = param.U_n(c_s_n_surf)
        ocp_p = param.U_p(c_s_p_surf)
        eta_r_n = int_curr_model.get_inverse_butler_volmer(j_n, j0_n, neg)
        eta_r_p = int_curr_model.get_inverse_butler_volmer(j_p, j0_p, pos)
        pot_model = pybamm.potential.Potential(param)
        ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n, ocp_p)
        eta_r_vars = pot_model.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
        self.variables.update({**ocp_vars, **eta_r_vars})

        # Electrolyte current
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        elyte_vars = eleclyte_current_model.get_explicit_leading_order(
            ocp_n, eta_r_n, i_local
        )
        self.variables.update(elyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        phi_e = self.variables["Electrolyte potential"]
        electrode_vars = electrode_model.get_explicit_leading_order(
            ocp_p, eta_r_p, phi_e
        )
        self.variables.update(electrode_vars)

        # Cut-off voltage
        voltage = self.variables["Terminal voltage"]
        self.events.append(voltage - param.voltage_low_cut)

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro", "1D micro")

    @property
    def default_geometry(self):
        if self.bc_options["dimensionality"] == 1:
            return pybamm.Geometry("1D macro", "1D micro")
        elif self.bc_options["dimensionality"] == 2:
            return pybamm.Geometry("1+1D macro", "1D micro")
        elif self.bc_options["dimensionality"] == 3:
            return pybamm.Geometry("2+1D macro", "1D micro")

   def set_boundary_conditions(self, bc_variables=None):
       # TO DO: edit below for SPM
        """Get boundary conditions"""
        # TODO: edit to allow constant-current and constant-power control
        param = self.set_of_parameters
        dimensionality = self.bc_options["dimensionality"]
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
