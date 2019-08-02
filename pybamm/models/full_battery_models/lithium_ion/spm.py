#
# Single Particle Model (SPM)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class SPM(BaseModel):
    """Single Particle Model (SPM) of a lithium-ion battery.

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, options=None, name="Single Particle Model"):
        super().__init__(options, name)

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

        if self.options["bc_options"]["dimensionality"] == 0:
            self.submodels["current collector"] = pybamm.current_collector.Uniform(
                self.param
            )
        elif self.options["bc_options"]["dimensionality"] == 1:
            raise NotImplementedError(
                "One-dimensional current collector submodel not implemented."
            )
        elif self.options["bc_options"]["dimensionality"] == 2:
            self.submodels[
                "current collector"
            ] = pybamm.current_collector.SingleParticlePotentialPair(self.param)

    def set_porosity_submodel(self):

        self.submodels["porosity"] = pybamm.porosity.Constant(self.param)

    def set_convection_submodel(self):

        self.submodels["convection"] = pybamm.convection.NoConvection(self.param)

    def set_interfacial_submodel(self):

        self.submodels[
            "negative interface"
        ] = pybamm.interface.lithium_ion.InverseButlerVolmer(self.param, "Negative")
        self.submodels[
            "positive interface"
        ] = pybamm.interface.lithium_ion.InverseButlerVolmer(self.param, "Positive")

    def set_particle_submodel(self):

        self.submodels["negative particle"] = pybamm.particle.fickian.SingleParticle(
            self.param, "Negative"
        )
        self.submodels["positive particle"] = pybamm.particle.fickian.SingleParticle(
            self.param, "Positive"
        )

    def set_negative_electrode_submodel(self):

        self.submodels["negative electrode"] = pybamm.electrode.ohm.LeadingOrder(
            self.param, "Negative"
        )

    def set_positive_electrode_submodel(self):

        self.submodels["positive electrode"] = pybamm.electrode.ohm.LeadingOrder(
            self.param, "Positive"
        )

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell

        self.submodels[
            "electrolyte conductivity"
        ] = electrolyte.conductivity.LeadingOrder(self.param)
        self.submodels[
            "electrolyte diffusion"
        ] = electrolyte.diffusion.ConstantConcentration(self.param)

    @property
    def default_geometry(self):
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality == 0:
            return pybamm.Geometry("1D macro", "1D micro")
        elif dimensionality == 1:
            return pybamm.Geometry("1+1D macro", "(1+0)+1D micro")
        elif dimensionality == 2:
            return pybamm.Geometry("2+1D macro", "(2+0)+1D micro")

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality == 0:
            return pybamm.ScipySolver()
        else:
            return pybamm.ScikitsDaeSolver()
