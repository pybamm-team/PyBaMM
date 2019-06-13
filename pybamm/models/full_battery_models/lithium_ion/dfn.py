#
# Doyle-Fuller-Newman (DFN) Model
#
import pybamm
from .base_lithium_ion_model import BaseModel


class DFN(BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery.
    **Extends:** :class:`pybamm.BaseLithiumIonModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "Doyle-Fuller-Newman model"
        self.param = pybamm.standard_parameters_lithium_ion

        self.submodels = {}

        self.set_current_collector_submodel()
        self.set_porosity_submodel()
        self.set_interfacial_submodel()
        self.set_particle_submodel()
        self.set_solid_submodel()
        self.set_electrolyte_submodel()
        self.set_thermal_submodel()

        self.build_model()

    def build_model(self):
        # TODO: put into base model

        # Set the fundamental variables
        for submodel in self.submodels.values():
            self.variables.update(submodel.get_fundamental_variables())

        # Set presolved variables
        for submodel in self.submodels.values():
            try:
                self.variables.update(submodel.get_coupled_variables(self.variables))
            except:
                submodel

        # Set model equations
        for submodel in self.submodels.values():
            submodel.set_rhs(self.variables)
            submodel.set_algebraic(self.variables)
            submodel.set_boundary_conditions(self.variables)
            submodel.set_initial_conditions(self.variables)
            self.update(submodel)

    def set_thermal_submodel(self):
        # TODO: put into base model

        if self.options["thermal"] is None:
            thermal_submodel = pybamm.thermal.Isothermal(self.param)
        elif self.options["thermal"] == "full":
            thermal_submodel = pybamm.thermal.FullModel(self.param)
        elif self.options["thermal"] == "lumped":
            thermal_submodel = pybamm.thermal.LumpedModel(self.param)
        else:
            raise KeyError("Unknown type of thermal model")

        self.submodels["thermal"] = thermal_submodel

    def set_current_collector_submodel(self):

        self.submodels["current collector"] = pybamm.current_collector.Uniform(
            self.param, "Negative"
        )

    def set_porosity_submodel(self):

        self.submodels["porosity"] = pybamm.porosity.Constant(self.param)

    def set_interfacial_submodel(self):

        self.submodels[
            "negative interface"
        ] = pybamm.interface.butler_volmer.LithiumIon(self.param, "Negative")
        self.submodels[
            "positive interface"
        ] = pybamm.interface.butler_volmer.LithiumIon(self.param, "Positive")

    def set_particle_submodel(self):

        self.submodels["negative particle"] = pybamm.particle.fickian.ManyParticles(
            self.param, "Negative"
        )
        self.submodels["positive particle"] = pybamm.particle.fickian.ManyParticles(
            self.param, "Positive"
        )

    def set_solid_submodel(self):

        self.submodels["negative electrode"] = pybamm.electrode.ohm.Full(
            self.param, "Negative"
        )
        self.submodels["positive electrode"] = pybamm.electrode.ohm.Full(
            self.param, "Positive"
        )

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell

        self.submodels["electrolyte conductivity"] = electrolyte.conductivity.FullModel(
            self.param
        )
        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.FullModel(
            self.param
        )

    #     "-----------------------------------------------------------------------------"
    #     "Parameters"
    #     param = pybamm.standard_parameters_lithium_ion
    #     i_boundary_cc = param.current_with_time
    #     self.variables["Current collector current density"] = i_boundary_cc

    #     "-----------------------------------------------------------------------------"
    #     "Model Variables"

    #     self.set_model_variables(param)

    #     c_e = self.variables["Electrolyte concentration"]
    #     c_s_n = self.variables["Negative particle concentration"]
    #     c_s_p = self.variables["Positive particle concentration"]
    #     ocp_n = self.variables["Negative electrode open circuit potential"]
    #     ocp_p = self.variables["Positive electrode open circuit potential"]
    #     eta_r_n = self.variables["Negative reaction overpotential"]
    #     eta_r_p = self.variables["Positive reaction overpotential"]

    #     "-----------------------------------------------------------------------------"
    #     "Submodels"
    #     # Exchange-current density
    #     c_e_n, _, c_e_p = c_e.orphans
    #     c_s_n_surf = pybamm.surf(c_s_n, set_domain=True)
    #     c_s_p_surf = pybamm.surf(c_s_p, set_domain=True)
    #     int_curr_model = pybamm.interface.LithiumIonReaction(param)

    #     # Interfacial current density
    #     j0_n = int_curr_model.get_exchange_current_densities(c_e_n, c_s_n_surf)
    #     j0_p = int_curr_model.get_exchange_current_densities(c_e_p, c_s_p_surf)
    #     j_n = int_curr_model.get_butler_volmer(j0_n, eta_r_n)
    #     j_p = int_curr_model.get_butler_volmer(j0_p, eta_r_p)
    #     j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
    #     self.variables.update(j_vars)

    #     # Particle models
    #     negative_particle_model = pybamm.particle.Standard(param)
    #     negative_particle_model.set_differential_system(c_s_n, j_n)
    #     positive_particle_model = pybamm.particle.Standard(param)
    #     positive_particle_model.set_differential_system(c_s_p, j_p)
    #     self.update(negative_particle_model)
    #     self.update(positive_particle_model)

    #     # Electrolyte concentration
    #     reactions = {
    #         "main": {"neg": {"s_plus": 1, "aj": j_n}, "pos": {"s_plus": 1, "aj": j_p}}
    #     }
    #     # Electrolyte diffusion model
    #     electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell
    # (param)
    #     electrolyte_diffusion_model.set_differential_system(self.variables, reactions)
    #     self.update(electrolyte_diffusion_model)

    #     eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell
    # (param)
    #     eleclyte_current_model.set_algebraic_system(self.variables, reactions)
    #     self.update(eleclyte_current_model)

    #     # Electrode models
    #     neg = ["negative electrode"]
    #     pos = ["positive electrode"]
    #     electrode_current_model = pybamm.electrode.Ohm(param)
    #     electrode_current_model.set_algebraic_system(self.variables, reactions, neg)
    #     electrode_current_model.set_algebraic_system(self.variables, reactions, pos)
    #     self.update(electrode_current_model)

    #     # Thermal model
    #     thermal_model = pybamm.thermal.Thermal(param)  # initialise empty submodel
    #     if self.options["thermal"] == "full":
    #         thermal_model.set_full_differential_system(self.variables, reactions)
    #     elif self.options["thermal"] == "lumped":
    #         thermal_model.set_x_lumped_differential_system(self.variables, reactions)
    #     self.update(thermal_model)

    #     "-----------------------------------------------------------------------------"
    #     "Post-process"

    #     # Exchange-current density
    #     j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
    #     self.variables.update(j_vars)

    #     # Potentials
    #     pot_model = pybamm.potential.Potential(param)
    #     ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n, ocp_p)
    #     eta_r_vars = pot_model.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
    #     self.variables.update({**ocp_vars, **eta_r_vars})

    #     # Voltage
    #     phi_s_n = self.variables["Negative electrode potential"]
    #     phi_s_p = self.variables["Positive electrode potential"]
    #     i_s_n = self.variables["Negative electrode current density"]
    #     i_s_p = self.variables["Positive electrode current density"]
    #     volt_vars = electrode_current_model.get_variables(
    #         phi_s_n, phi_s_p, i_s_n, i_s_p
    #     )
    #     self.variables.update(volt_vars)

    #     # Cut-off voltage
    #     voltage = self.variables["Terminal voltage"]
    #     self.events.append(voltage - param.voltage_low_cut)

    # def set_model_variables(self, param):
    #     c_s_n = pybamm.standard_variables.c_s_n
    #     c_s_p = pybamm.standard_variables.c_s_p
    #     c_e = pybamm.standard_variables.c_e
    #     phi_e = pybamm.standard_variables.phi_e
    #     phi_s_p = pybamm.standard_variables.phi_s_p
    #     phi_s_n = pybamm.standard_variables.phi_s_n
    #     delta_phi_n = phi_s_n - phi_e.orphans[0]
    #     delta_phi_p = phi_s_p - phi_e.orphans[2]

    #     c_s_n_surf = pybamm.surf(c_s_n, set_domain=True)
    #     c_s_p_surf = pybamm.surf(c_s_p, set_domain=True)
    #     ocp_n = param.U_n(c_s_n_surf)
    #     ocp_p = param.U_p(c_s_p_surf)
    #     eta_r_n = delta_phi_n - ocp_n
    #     eta_r_p = delta_phi_p - ocp_p

    #     self.variables.update(
    #         {
    #             "Electrolyte concentration": c_e,
    #             "Negative particle concentration": c_s_n,
    #             "Positive particle concentration": c_s_p,
    #             "Electrolyte potential": phi_e,
    #             "Negative electrode potential": phi_s_n,
    #             "Positive electrode potential": phi_s_p,
    #             "Negative electrode surface potential difference": delta_phi_n,
    #             "Positive electrode surface potential difference": delta_phi_p,
    #             "Negative electrode open circuit potential": ocp_n,
    #             "Positive electrode open circuit potential": ocp_p,
    #             "Negative reaction overpotential": eta_r_n,
    #             "Positive reaction overpotential": eta_r_p,
    #         }
    #     )

    #     if self.options["thermal"] == "full":
    #         self.variables.update({"Cell temperature": pybamm.standard_variables.T})
    #     if self.options["thermal"] == "lumped":
    #         T_av = pybamm.standard_variables.T_av
    #         T_n = pybamm.Broadcast(T_av, ["negative electrode"])
    #         T_s = pybamm.Broadcast(T_av, ["separator"])
    #         T_p = pybamm.Broadcast(T_av, ["positive electrode"])
    #         T = pybamm.Concatenation(T_n, T_s, T_p)
    #         self.variables.update(
    #             {"Average cell temperature": T_av, "Cell temperature": T}
    #         )

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
