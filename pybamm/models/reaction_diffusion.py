#
# Reaction-diffusion model
#
import pybamm


class ReactionDiffusionModel(pybamm.BaseBatteryModel):
    """Reaction-diffusion model.

    **Extends**: :class:`pybamm.BaseBatteryModel`

    """

    def __init__(self, options=None):
        super().__init__(options, "Reaction diffusion model")
        # NOTE: set_standard_output_variables sets the lead acid timescales,
        # so if paramaters are changed here the timescale in the method
        # set_standard_output_variables may need to be altered
        self.param = pybamm.standard_parameters_lead_acid

        # Manually set porosity parameters
        self.param.epsilon_n = pybamm.Scalar(1)
        self.param.epsilon_s = pybamm.Scalar(1)
        self.param.epsilon_p = pybamm.Scalar(1)

        self.set_thermal_submodel()
        self.set_reactions()
        self.set_current_collector_submodel()
        self.set_convection_submodel()
        self.set_porosity_submodel()
        self.set_interfacial_submodel()
        self.set_electrolyte_submodel()

        self.build_model()

    def set_thermal_submodel(self):
        self.submodels["thermal"] = pybamm.thermal.Isothermal(self.param)

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
            self.param, self.reactions
        )

    def set_interfacial_submodel(self):
        self.submodels[
            "negative interface"
        ] = pybamm.interface.lead_acid.InverseButlerVolmer(self.param, "Negative")
        self.submodels[
            "positive interface"
        ] = pybamm.interface.lead_acid.InverseButlerVolmer(self.param, "Positive")

    def set_voltage_variables(self):
        "overwrite to set nothing"
        return None

    @property
    def default_parameter_values(self):
        return pybamm.lead_acid.BaseModel().default_parameter_values

    def set_standard_output_variables(self):
        super().set_standard_output_variables()
        # Set current variables to use lead acid timescale
        icell = pybamm.standard_parameters_lead_acid.current_with_time
        icell_dim = (
            pybamm.standard_parameters_lead_acid.dimensional_current_density_with_time
        )
        I = pybamm.standard_parameters_lead_acid.dimensional_current_with_time
        self.variables.update(
            {
                "Total current density": icell,
                "Total current density [A.m-2]": icell_dim,
                "Current [A]": I,
            }
        )

        # Set time variables to use lead acid timescale
        time_scale = pybamm.standard_parameters_lead_acid.tau_discharge
        self.variables.update(
            {
                "Time [s]": pybamm.t * time_scale,
                "Time [min]": pybamm.t * time_scale / 60,
                "Time [h]": pybamm.t * time_scale / 3600,
                "Discharge capacity [A.h]": I * pybamm.t * time_scale / 3600,
            }
        )

    def set_reactions(self):

        # Should probably refactor as this is a bit clunky at the moment
        # Maybe each reaction as a Reaction class so we can just list names of classes
        param = self.param
        icd = " interfacial current density"
        self.reactions = {
            "main": {
                "Negative": {"s": param.s_n, "aj": "Negative electrode" + icd},
                "Positive": {"s": param.s_p, "aj": "Positive electrode" + icd},
            }
        }
