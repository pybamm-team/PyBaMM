#
# Doyle-Fuller-Newman (DFN) Model
#
import pybamm
from .base_lithium_ion_model import BaseModel


class DFN(BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery.


    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, options=None, name="Doyle-Fuller-Newman model"):
        super().__init__(options, name)

        self.set_reactions()
        self.set_porosity_submodel()
        self.set_convection_submodel()
        self.set_interfacial_submodel()
        self.set_particle_submodel()
        self.set_solid_submodel()
        self.set_electrolyte_submodel()
        self.set_thermal_submodel()
        self.set_current_collector_submodel()

        self.build_model()

    def set_porosity_submodel(self):

        self.submodels["porosity"] = pybamm.porosity.Constant(self.param)

    def set_convection_submodel(self):

        self.submodels[
            "through-cell convection"
        ] = pybamm.convection.through_cell.NoConvection(self.param)

    def set_interfacial_submodel(self):

        self.submodels[
            "negative interface"
        ] = pybamm.interface.lithium_ion.ButlerVolmer(self.param, "Negative")
        self.submodels[
            "positive interface"
        ] = pybamm.interface.lithium_ion.ButlerVolmer(self.param, "Positive")

    def set_particle_submodel(self):

        self.submodels["negative particle"] = pybamm.particle.fickian.ManyParticles(
            self.param, "Negative"
        )
        self.submodels["positive particle"] = pybamm.particle.fickian.ManyParticles(
            self.param, "Positive"
        )

    def set_solid_submodel(self):

        self.submodels["negative electrode"] = pybamm.electrode.ohm.Full(
            self.param, "Negative", self.reactions
        )
        self.submodels["positive electrode"] = pybamm.electrode.ohm.Full(
            self.param, "Positive", self.reactions
        )

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell

        self.submodels["electrolyte conductivity"] = electrolyte.conductivity.Full(
            self.param, self.reactions
        )
        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.Full(
            self.param, self.reactions
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
