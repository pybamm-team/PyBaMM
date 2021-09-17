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
           derivation of a single particle model with electrolyte”. Journal of The
           Electrochemical Society, 166(15):A3693–A3706, 2019

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(
        self, options=None, name="Single Particle Model with electrolyte", build=True
    ):
        super().__init__(options, name)
        # For degradation models we use the "x-average" form since this is a
        # reduced-order model with uniform current density in the electrodes
        self.x_average = True

        self.set_external_circuit_submodel()
        self.set_porosity_submodel()
        self.set_crack_submodel()
        self.set_active_material_submodel()
        self.set_tortuosity_submodels()
        self.set_convection_submodel()
        self.set_interfacial_submodel()
        self.set_other_reaction_submodels_to_zero()
        self.set_particle_submodel()
        self.set_negative_electrode_submodel()
        self.set_electrolyte_submodel()
        self.set_positive_electrode_submodel()
        self.set_thermal_submodel()
        self.set_current_collector_submodel()
        self.set_sei_submodel()
        self.set_lithium_plating_submodel()

        if build:
            self.build_model()

        pybamm.citations.register("Marquis2019")

    def set_convection_submodel(self):

        self.submodels[
            "through-cell convection"
        ] = pybamm.convection.through_cell.NoConvection(self.param)
        self.submodels[
            "transverse convection"
        ] = pybamm.convection.transverse.NoConvection(self.param)

    def set_tortuosity_submodels(self):
        self.submodels["electrolyte tortuosity"] = pybamm.tortuosity.Bruggeman(
            self.param, "Electrolyte", self.options, True
        )
        self.submodels["electrode tortuosity"] = pybamm.tortuosity.Bruggeman(
            self.param, "Electrode", self.options, True
        )

    def set_interfacial_submodel(self):

        if self.options["surface form"] == "false":
            self.submodels["negative interface"] = pybamm.interface.InverseButlerVolmer(
                self.param, "Negative", "lithium-ion main", self.options
            )
            self.submodels["positive interface"] = pybamm.interface.InverseButlerVolmer(
                self.param, "Positive", "lithium-ion main", self.options
            )
            self.submodels[
                "negative interface current"
            ] = pybamm.interface.CurrentForInverseButlerVolmer(
                self.param, "Negative", "lithium-ion main"
            )
            self.submodels[
                "positive interface current"
            ] = pybamm.interface.CurrentForInverseButlerVolmer(
                self.param, "Positive", "lithium-ion main"
            )
        else:
            self.submodels["negative interface"] = pybamm.interface.ButlerVolmer(
                self.param, "Negative", "lithium-ion main", self.options
            )

            self.submodels["positive interface"] = pybamm.interface.ButlerVolmer(
                self.param, "Positive", "lithium-ion main", self.options
            )

    def set_particle_submodel(self):

        if isinstance(self.options["particle"], str):
            particle_left = self.options["particle"]
            particle_right = self.options["particle"]
        else:
            particle_left, particle_right = self.options["particle"]
        for particle_side, domain in [
            [particle_left, "Negative"],
            [particle_right, "Positive"],
        ]:
            if particle_side == "Fickian diffusion":
                self.submodels[
                    domain.lower() + " particle"
                ] = pybamm.particle.no_distribution.XAveragedFickianDiffusion(
                    self.param, domain
                )
            elif particle_side in [
                "uniform profile",
                "quadratic profile",
                "quartic profile",
            ]:
                self.submodels[
                    domain.lower() + " particle"
                ] = pybamm.particle.no_distribution.XAveragedPolynomialProfile(
                    self.param, domain, particle_side
                )

    def set_negative_electrode_submodel(self):

        self.submodels["negative electrode potential"] = pybamm.electrode.ohm.Composite(
            self.param, "Negative"
        )

    def set_positive_electrode_submodel(self):

        self.submodels["positive electrode potential"] = pybamm.electrode.ohm.Composite(
            self.param, "Positive"
        )

    def set_electrolyte_submodel(self):

        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        if self.options["electrolyte conductivity"] not in [
            "default",
            "composite",
            "integrated",
        ]:
            raise pybamm.OptionError(
                "electrolyte conductivity '{}' not suitable for SPMe".format(
                    self.options["electrolyte conductivity"]
                )
            )

        if self.options["surface form"] == "false":
            if self.options["electrolyte conductivity"] in ["default", "composite"]:
                self.submodels[
                    "electrolyte conductivity"
                ] = pybamm.electrolyte_conductivity.Composite(self.param)
            elif self.options["electrolyte conductivity"] == "integrated":
                self.submodels[
                    "electrolyte conductivity"
                ] = pybamm.electrolyte_conductivity.Integrated(self.param)
        elif self.options["surface form"] == "differential":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.CompositeDifferential(self.param, domain)
        elif self.options["surface form"] == "algebraic":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.CompositeAlgebraic(self.param, domain)

        self.submodels["electrolyte diffusion"] = pybamm.electrolyte_diffusion.Full(
            self.param
        )
