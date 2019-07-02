#
# Reaction-diffusion model
#
import pybamm


class ReactionDiffusionModel(pybamm.BaseBatteryModel):
    """Reaction-diffusion model.

    **Extends**: :class:`pybamm.BaseBatteryModel`

    """

    def __init__(self, options=None):

        if not options:
            options = {}

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
        self.set_convection_submodel()
        self.set_porosity_submodel()
        self.set_interfacial_submodel()
        self.set_electrolyte_submodel()

        self.build_model()

    def set_current_collector_submodel(self):

        self.submodels["current collector"] = pybamm.current_collector.Uniform(
            self.param
        )

    def set_porosity_submodel(self):
        self.submodels["porosity"] = pybamm.porosity.Constant(self.param)

    def set_convection_submodel(self):
        self.submodels["convection"] = pybamm.convection.NoConvection(self.param)

    def set_electrolyte_submodel(self):
        electrolyte = pybamm.electrolyte.stefan_maxwell
        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.Full(
            self.param, ocp=True
        )

    def set_interfacial_submodel(self):
        self.submodels[
            "negative interface"
        ] = pybamm.interface.inverse_butler_volmer.LeadAcid(self.param, "Negative")
        self.submodels[
            "positive interface"
        ] = pybamm.interface.inverse_butler_volmer.LeadAcid(self.param, "Positive")

    def set_voltage_variables(self):
        "overwrite to set nothing"
        return None

    @property
    def default_parameter_values(self):
        return pybamm.lead_acid.BaseModel().default_parameter_values
