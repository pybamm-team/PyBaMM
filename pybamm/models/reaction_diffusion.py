#
# Reaction-diffusion model
#
import pybamm


class ReactionDiffusionModel(pybamm.BaseBatteryModel):
    """Reaction-diffusion model.

    **Extends**: :class:`pybamm.BaseBatteryModel`

    """

    def __init__(self, options={}):
        options.update(
            {"Voltage": "Off"}
        )  # annoying option only for reaction diffusion
        super().__init__(options)
        self.name = "Reaction diffusion model"
        self.param = pybamm.standard_parameters_lead_acid

        # Manually set porosity parameters
        self.param.epsilon_n = pybamm.Scalar(1)
        self.param.epsilon_s = pybamm.Scalar(1)
        self.param.epsilon_p = pybamm.Scalar(1)

        self.set_current_collector_submodel()
        self.set_porosity_submodel()
        self.set_interfacial_submodel()
        self.set_electrolyte_submodel()

        self.build_model()

        # "-----------------------------------------------------------------------------"
        # "Parameters"
        # current = param.current_with_time

        # "-----------------------------------------------------------------------------"
        # "Model Variables"

        # c_e = pybamm.standard_variables.c_e

        # "-----------------------------------------------------------------------------"
        # "Submodels"

        # # Interfacial current density
        # neg = ["negative electrode"]
        # pos = ["positive electrode"]
        # negative_interface = pybamm.interface.inverse_bulter_volmer.LeadAcid(
        #     param, "Negative"
        # )
        # positive_interface = pybamm.interface.inverse_bulter_volmer.LeadAcid(
        #     param, "Negative"
        # )

        # j_n = int_curr_model.get_homogeneous_interfacial_current(current, neg)
        # j_p = int_curr_model.get_homogeneous_interfacial_current(current, pos)

        # # Porosity
        # epsilon = pybamm.Scalar(1)

        # # Electrolyte concentration
        # j_n = pybamm.Broadcast(j_n, neg)
        # j_p = pybamm.Broadcast(j_p, pos)
        # self.variables = {"Electrolyte concentration": c_e, "Porosity": epsilon}
        # reactions = {
        #     "main": {
        #         "neg": {"s_plus": 1, "aj": j_n},
        #         "pos": {"s_plus": 1, "aj": j_p},
        #         "porosity change": 0,
        #     }
        # }
        # eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        # eleclyte_conc_model.set_differential_system(self.variables, reactions)
        # self.update(eleclyte_conc_model)

        # "-----------------------------------------------------------------------------"
        # "Post-Processing"

        # # Exchange-current density
        # c_e_n, _, c_e_p = c_e.orphans
        # j0_n = int_curr_model.get_exchange_current_densities(c_e_n, neg)
        # j0_p = int_curr_model.get_exchange_current_densities(c_e_p, pos)
        # j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        # self.variables.update(j_vars)

    def set_current_collector_submodel(self):

        self.submodels["current collector"] = pybamm.current_collector.Uniform(
            self.param, "Negative"
        )

    def set_porosity_submodel(self):
        self.submodels["porosity"] = pybamm.porosity.Constant(self.param)

    def set_electrolyte_submodel(self):
        electrolyte = pybamm.electrolyte.stefan_maxwell
        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.FullModel(
            self.param
        )

    def set_interfacial_submodel(self):
        self.submodels[
            "negative interface"
        ] = pybamm.interface.inverse_butler_volmer.LeadAcid(self.param, "Negative")
        self.submodels[
            "positive interface"
        ] = pybamm.interface.inverse_butler_volmer.LeadAcid(self.param, "Positive")

    @property
    def default_parameter_values(self):
        return pybamm.lead_acid.BaseModel().default_parameter_values
