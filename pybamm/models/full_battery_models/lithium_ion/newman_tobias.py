#
# Newman Tobias Model
#
import pybamm
from .base_lithium_ion_model import BaseModel


class NewmanTobias(BaseModel):
    """
    Newman-Tobias model of a lithium-ion battery based on the formulation in [1]_.
    This model assumes a uniform concentration profile in the electrolyte.
    Unlike the model posed in [1]_, this models accounts for nonlinear Butler-Volmer
    kinetics, and tracks the average concentration in the solid phase in each electrode.
    This is analagous to including an equation for the state of charge as in [2]_.

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
    .. [1] JS Newman and CW Tobias. "Theoretical Analysis of Current Distribution
           in Porous Electrodes". Journal of The Electrochemical Society,
           109(12):A1183-A1191, 1962
    .. [2] HN Chu, SU Kim, SK Rahimian, JB Siegel and CW Monroe. "Parameterization
           of prismatic lithium–iron–phosphate cells through a streamlined
           thermal/electrochemical model". Journal of Power Sources, 453, p.227787,
           2020

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, options=None, name="Newman-Tobias model", build=True):

        options = options or {}
        # set option to "uniform profile" if not provided
        if "particle" not in options:
            options["particle"] = "uniform profile"
        # raise error if any other particle option is selected
        if options["particle"] != "uniform profile":
            raise pybamm.OptionError(
                "Newman-Tobias model cannot model mass transport within the particles. "
                "The 'particle' option must be 'uniform profile' but is {}.".format(
                    options["particle"]
                )
            )
        # currently not available as a "2+1D" model (see #1399)
        dimensionality_option = options.get(
            "dimensionality", "none"
        )  # return "none" if option not given
        if dimensionality_option == 2:
            raise pybamm.OptionError(
                "Newman-Tobias model does not current support 2D current collectors"
            )
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

        pybamm.citations.register("Newman1962")
        pybamm.citations.register("Chu2020")

    def set_porosity_submodel(self):

        if self.options["sei porosity change"] == "false":
            self.submodels["porosity"] = pybamm.porosity.Constant(self.param)
        elif self.options["sei porosity change"] == "true":
            self.submodels["porosity"] = pybamm.porosity.Full(self.param)

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

        self.submodels["negative particle"] = pybamm.particle.PolynomialSingleParticle(
            self.param, "Negative", "uniform profile"
        )
        self.submodels["positive particle"] = pybamm.particle.PolynomialSingleParticle(
            self.param, "Positive", "uniform profile"
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

        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte_diffusion.ConstantConcentration(self.param)

        if self.options["electrolyte conductivity"] not in ["default", "full"]:
            raise pybamm.OptionError(
                "electrolyte conductivity '{}' not suitable for Newman-Tobias".format(
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
