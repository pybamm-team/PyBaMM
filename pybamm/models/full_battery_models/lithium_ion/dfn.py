#
# Doyle-Fuller-Newman (DFN) Model
#
import pybamm
from .base_lithium_ion_model import BaseModel


class DFN(BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery, from [1]_.

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
           derivation of a single particle model with electrolyte”. Journal of The
           Electrochemical Society, 166(15):A3693–A3706, 2019


    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, options=None, name="Doyle-Fuller-Newman model", build=True):
        super().__init__(options, name)

        self.set_external_circuit_submodel()
        self.set_porosity_submodel()
        self.set_crack_submodel()
        self.set_active_material_submodel()
        self.set_tortuosity_submodels()
        self.set_convection_submodel()
        self.set_interfacial_submodel()
        self.set_other_reaction_submodels_to_zero()
        self.set_particle_submodel()
        self.set_solid_submodel()
        self.set_electrolyte_submodel()
        self.set_thermal_submodel()
        self.set_current_collector_submodel()
        self.set_sei_submodel()
        self.set_lithium_plating_submodel()

        if build:
            self.build_model()

        pybamm.citations.register("Doyle1993")

    def set_porosity_submodel(self):

        if (
            self.options["SEI porosity change"] == "false"
            and self.options["lithium plating porosity change"] == "false"
        ):
            self.submodels["porosity"] = pybamm.porosity.Constant(
                self.param, self.options
            )
        elif (
            self.options["SEI porosity change"] == "true"
            or self.options["lithium plating porosity change"] == "true"
        ):
            self.submodels["porosity"] = pybamm.porosity.Full(self.param, self.options)

    def set_active_material_submodel(self):

        if self.options["loss of active material"] == "none":
            self.submodels[
                "negative active material"
            ] = pybamm.active_material.Constant(self.param, "Negative", self.options)
            self.submodels[
                "positive active material"
            ] = pybamm.active_material.Constant(self.param, "Positive", self.options)
        elif self.options["loss of active material"] == "both":
            self.submodels[
                "negative active material"
            ] = pybamm.active_material.VaryingFull(self.param, "Negative", self.options)
            self.submodels[
                "positive active material"
            ] = pybamm.active_material.VaryingFull(self.param, "Positive", self.options)
        elif self.options["loss of active material"] == "negative":
            self.submodels[
                "negative active material"
            ] = pybamm.active_material.VaryingFull(self.param, "Negative", self.options)
            self.submodels[
                "positive active material"
            ] = pybamm.active_material.Constant(self.param, "Positive", self.options)
        elif self.options["loss of active material"] == "positive":
            self.submodels[
                "negative active material"
            ] = pybamm.active_material.Constant(self.param, "Negative", self.options)
            self.submodels[
                "positive active material"
            ] = pybamm.active_material.VaryingFull(self.param, "Positive", self.options)

    def set_convection_submodel(self):

        self.submodels[
            "transverse convection"
        ] = pybamm.convection.transverse.NoConvection(self.param)
        self.submodels[
            "through-cell convection"
        ] = pybamm.convection.through_cell.NoConvection(self.param)

    def set_interfacial_submodel(self):

        self.submodels["negative interface"] = pybamm.interface.ButlerVolmer(
            self.param, "Negative", "lithium-ion main", self.options
        )
        self.submodels["positive interface"] = pybamm.interface.ButlerVolmer(
            self.param, "Positive", "lithium-ion main", self.options
        )

    def set_particle_submodel(self):

        if self.options["particle"] == "Fickian diffusion":
            self.submodels["negative particle"] = pybamm.particle.FickianManyParticles(
                self.param, "Negative"
            )
            self.submodels["positive particle"] = pybamm.particle.FickianManyParticles(
                self.param, "Positive"
            )
        elif self.options["particle"] in [
            "uniform profile",
            "quadratic profile",
            "quartic profile",
        ]:
            self.submodels[
                "negative particle"
            ] = pybamm.particle.PolynomialManyParticles(
                self.param, "Negative", self.options["particle"]
            )
            self.submodels[
                "positive particle"
            ] = pybamm.particle.PolynomialManyParticles(
                self.param, "Positive", self.options["particle"]
            )

    def set_solid_submodel(self):

        if self.options["surface form"] == "false":
            submod_n = pybamm.electrode.ohm.Full(self.param, "Negative")
            submod_p = pybamm.electrode.ohm.Full(self.param, "Positive")
        else:
            submod_n = pybamm.electrode.ohm.SurfaceForm(self.param, "Negative")
            submod_p = pybamm.electrode.ohm.SurfaceForm(self.param, "Positive")

        self.submodels["negative electrode potential"] = submod_n
        self.submodels["positive electrode potential"] = submod_p

    def set_electrolyte_submodel(self):

        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        self.submodels["electrolyte diffusion"] = pybamm.electrolyte_diffusion.Full(
            self.param
        )

        if self.options["electrolyte conductivity"] not in ["default", "full"]:
            raise pybamm.OptionError(
                "electrolyte conductivity '{}' not suitable for DFN".format(
                    self.options["electrolyte conductivity"]
                )
            )

        if self.options["surface form"] == "false":
            self.submodels[
                "electrolyte conductivity"
            ] = pybamm.electrolyte_conductivity.Full(self.param)
        elif self.options["surface form"] == "differential":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.FullDifferential(self.param, domain)
        elif self.options["surface form"] == "algebraic":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.FullAlgebraic(self.param, domain)
