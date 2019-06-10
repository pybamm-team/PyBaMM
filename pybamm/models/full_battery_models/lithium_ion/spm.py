#
# Single Particle Model (SPM)
#
import pybamm


class SPM(pybamm.BaseLithiumIonModel):
    """Single Particle Model (SPM) of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        # TODO: set self.submodel in base class
        self.submodels = {}

        self.name = "Single particle model"
        self.param = pybamm.standard_parameters_lithium_ion

        # Initialise submodels
        self.submodels["thermal"] = pybamm.thermal.Isothermal(self.param)
        self.submodels["negative particle"] = pybamm.particle.Standard(self.param)
        self.submodels["positive particle"] = pybamm.particle.Standard(self.param)
        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte_diffusion.StefanMaxwell(self.param)
        self.submodels[
            "electrolyte current"
        ] = pybamm.electrolyte_current.MacInnesStefanMaxwell(self.param)

        # Create model
        self.create_model()

        # Events
        # Cut-off voltage
        voltage = self.variables["Terminal voltage"]
        self.events.append(voltage - self.param.voltage_low_cut)

        # self.set_current_collector_submodel()
        # self.set_interfacial_submodel()
        # self.set_particle_submodel()
        # self.set_solid_submodel()
        # self.set_electrolyte_submodel()
        # self.set_thermal_submodel()

    def create_model(self):
        # TODO: put into base model

        # Set the fundamental variables
        for submodel in self.submodels.values():
            self.variables.update(submodel.get_fundamental_variables(self.variables))

        # Set presolved variables
        for submodel in self.submodels.values():
            self.variables.update(submodel.get_derived_variables(self.variables))

        # Set model equations
        for submodel in self.submodels.values():
            submodel.set_rhs(self.variables)
            submodel.set_algebraic(self.variables)
            submodel.set_boundary_conditions(self.variables)
            submodel.set_initial_conditions(self.variables)
            self.update(submodel)

    def set_thermal_model(self):
        # TODO: put into base model

        if self.options["thermal"] is None:
            thermal_submodel = pybamm.IsothermalSubModel()
        elif self.options["thermal"] == "full":
            thermal_submodel = pybamm.FullThermalSubModel()
        elif self.options["thermal"] == "lumped":
            thermal_submodel = pybamm.LumpedThermalSubmodel()
        else:
            raise KeyError("Unknown type of thermal model")

        self.submodels["thermal"] = thermal_submodel

    def set_current_collector_submodel(self):
        # TODO: put into base model

        # this is where the fast conductivity limit which set the 1D bc for the
        # problem should go
        if self.options["current collector"] is None:
            self.submodels[
                "negative current collector"
            ] = pybamm.current_collector.Fast(self.param)
            self.submodels[
                "positive current collector"
            ] = pybamm.current_collector.Fast(self.param)
        elif self.options["current collector"] == "ohm":
            self.submodels["negative current collector"] = pybamm.current_collector.Ohm(
                self.param
            )
            self.submodels["positive current collector"] = pybamm.current_collector.Ohm(
                self.param
            )

    def set_interfacial_submodel(self):
        self.submodels["interface"] = pybamm.interface.LithiumIonReaction(self.param)

    def set_particle_submodel(self):
        self.submodels["negative particle"] = pybamm.particle.Standard(self.param)
        self.submodels["positive particle"] = pybamm.particle.Standard(self.param)

    def set_solid_submodel(self):
        self.submodels["negative solid"] = pybamm.electrode.Ohm(self.param)
        self.submodels["positive solid"] = pybamm.electrode.Ohm(self.param)

    def set_electrolyte_submodel(self):
        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte_diffusion.StefanMaxwell(self.param)

        self.submodels[
            "electrolyte current"
        ] = pybamm.electrolyte_current.MacInnesStefanMaxwell(self.param)

    def why(self, options=None):
        super().__init__(options)
        self.name = "Single Particle Model"

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lithium_ion
        i_boundary_cc = param.current_with_time
        self.variables["Current collector current density"] = i_boundary_cc

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_s_n = pybamm.standard_variables.c_s_n
        c_s_p = pybamm.standard_variables.c_s_p

        self.variables.update(
            {
                "Negative particle concentration": c_s_n,
                "Positive particle concentration": c_s_p,
            }
        )

        if self.options["thermal"] == "full":
            self.variables.update({"Cell temperature": pybamm.standard_variables.T})
        if self.options["thermal"] == "lumped":
            self.variables.update(
                {"Average cell temperature": pybamm.standard_variables.T_av}
            )

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
        self.update(negative_particle_model, positive_particle_model)

        # Thermal model
        thermal_model = pybamm.thermal.Thermal(param)  # initialise empty submodel
        if self.options["thermal"] == "full":
            thermal_model.set_full_differential_system()
        elif self.options["thermal"] == "lumped":
            thermal_model.set_x_lumped_differential_system()
        self.update(thermal_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Electrolyte concentration
        c_e = pybamm.Scalar(1)
        N_e = pybamm.Scalar(0)
        electrolyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        conc_vars = electrolyte_conc_model.get_variables(c_e, N_e)
        self.variables.update(conc_vars)

        # Exchange-current density
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
        elyte_vars = eleclyte_current_model.get_explicit_leading_order(self.variables)
        self.variables.update(elyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        electrode_vars = electrode_model.get_explicit_leading_order(self.variables)
        self.variables.update(electrode_vars)

        # Cut-off voltage
        voltage = self.variables["Terminal voltage"]
        self.events.append(voltage - param.voltage_low_cut)

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro", "1D micro")
