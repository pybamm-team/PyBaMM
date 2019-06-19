#
# Single Particle Model with Electrolyte (SPMe)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class SPMe(BaseModel):
    """Single Particle Model with Electrolyte (SPMe) of a lithium-ion battery.
    **Extends:** :class:`pybamm.BaseLithiumIonModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "Single Particle Model with electrolyte"

        self.set_current_collector_submodel()
        self.set_porosity_submodel()
        self.set_convection_submodel()
        self.set_interfacial_submodel()
        self.set_particle_submodel()
        self.set_negative_electrode_submodel()
        self.set_electrolyte_submodel()
        self.set_positive_electrode_submodel()
        self.set_thermal_submodel()

        self.build_model()

    def set_current_collector_submodel(self):

        self.submodels["current collector"] = pybamm.current_collector.Uniform(
            self.param, "Negative"
        )

    def set_porosity_submodel(self):

        self.submodels["porosity"] = pybamm.porosity.Constant(self.param)

    def set_convection_submodel(self):

        self.submodels["convection"] = pybamm.convection.NoConvection(self.param)

    def set_interfacial_submodel(self):

        self.submodels[
            "negative interface"
        ] = pybamm.interface.inverse_bulter_volmer.LithiumIon(self.param, "Negative")
        self.submodels[
            "positive interface"
        ] = pybamm.interface.inverse_bulter_volmer.LithiumIon(self.param, "Positive")

    def set_particle_submodel(self):

        self.submodels["negative particle"] = pybamm.particle.fickian.SingleParticle(
            self.param, "Negative"
        )
        self.submodels["positive particle"] = pybamm.particle.fickian.SingleParticle(
            self.param, "Positive"
        )

    def set_negative_electrode_submodel(self):

        self.submodels["negative electrode"] = pybamm.electrode.ohm.Combined(
            self.param, "Negative"
        )

    def set_positive_electrode_submodel(self):

        self.submodels["positive electrode"] = pybamm.electrode.ohm.Combined(
            self.param, "Positive"
        )

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell

        self.submodels[
            "electrolyte conductivity"
        ] = electrolyte.conductivity.CombinedOrderModel(self.param)
        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.FullModel(
            self.param
        )

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro", "1D micro")
