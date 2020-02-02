#
# Single Particle Model with Electrolyte (SPMe)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class SPMe(BaseModel):
    """Single Particle Model with Electrolyte (SPMe) of a lithium-ion battery, from
    [1]_.

    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model.
    name : str, optional
        The name of the model.
    build :  bool, optional
        Whether to build the model on instantiation. Default is True. Setting this
        option to False allows users to change any number of the submodels before
        building the complete model (submodels cannot be changed after the model is
        built).

    References
    ----------
    .. [1] SG Marquis, V Sulzer, R Timms, CP Please and SJ Chapman. “An asymptotic
           derivation of a single particle model with electrolyte”. In: arXiv preprint
           arXiv:1905.12553 (2019).

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(
        self, options=None, name="Single Particle Model with electrolyte", build=True
    ):
        super().__init__(options, name)

        self.set_external_circuit_submodel()
        self.set_reactions()
        self.set_porosity_submodel()
        self.set_tortuosity_submodels()
        self.set_convection_submodel()
        self.set_interfacial_submodel()
        self.set_particle_submodel()
        self.set_negative_electrode_submodel()
        self.set_electrolyte_submodel()
        self.set_positive_electrode_submodel()
        self.set_thermal_submodel()
        self.set_current_collector_submodel()

        if build:
            self.build_model()

    def set_porosity_submodel(self):

        self.submodels["porosity"] = pybamm.porosity.Constant(self.param)

    def set_tortuosity_submodels(self):
        self.submodels["electrolyte tortuosity"] = pybamm.tortuosity.Bruggeman(
            self.param, "Electrolyte", True
        )
        self.submodels["electrode tortuosity"] = pybamm.tortuosity.Bruggeman(
            self.param, "Electrode", True
        )

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

        if self.options["particle"] == "Fickian diffusion":
            self.submodels[
                "negative particle"
            ] = pybamm.particle.fickian.SingleParticle(self.param, "Negative")
            self.submodels[
                "positive particle"
            ] = pybamm.particle.fickian.SingleParticle(self.param, "Positive")
        elif self.options["particle"] == "fast diffusion":
            self.submodels["negative particle"] = pybamm.particle.fast.SingleParticle(
                self.param, "Negative"
            )
            self.submodels["positive particle"] = pybamm.particle.fast.SingleParticle(
                self.param, "Positive"
            )

    def set_negative_electrode_submodel(self):

        self.submodels["negative electrode"] = pybamm.electrode.ohm.Composite(
            self.param, "Negative"
        )

    def set_positive_electrode_submodel(self):

        self.submodels["positive electrode"] = pybamm.electrode.ohm.Composite(
            self.param, "Positive"
        )

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell

        self.submodels["electrolyte conductivity"] = electrolyte.conductivity.Composite(
            self.param
        )
        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.Full(
            self.param, self.reactions
        )

    @property
    def default_geometry(self):
        dimensionality = self.options["dimensionality"]
        if dimensionality == 0:
            return pybamm.Geometry("1D macro", "1D micro")
        elif dimensionality == 1:
            return pybamm.Geometry("1+1D macro", "(1+0)+1D micro")
        elif dimensionality == 2:
            return pybamm.Geometry("2+1D macro", "(2+0)+1D micro")
