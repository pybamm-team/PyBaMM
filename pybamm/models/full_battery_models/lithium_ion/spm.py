#
# Single Particle Model (SPM)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class SPM(BaseModel):
    """Single Particle Model (SPM) of a lithium-ion battery, from [1]_.

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

    def __init__(self, options=None, name="Single Particle Model", build=True):
        # Use 'algebraic' surface form if non-default kinetics are used
        options = options or {}
        kinetics = options.get("intercalation kinetics")
        surface_form = options.get("surface form")
        if kinetics is not None and surface_form is None:
            options["surface form"] = "algebraic"
        super().__init__(options, name)
        # For degradation models we use the "x-average" form since this is a
        # reduced-order model with uniform current density in the electrodes
        self.x_average = True

        self.set_external_circuit_submodel()
        self.set_porosity_submodel()
        self.set_interface_utilisation_submodel()
        self.set_crack_submodel()
        self.set_active_material_submodel()
        self.set_tortuosity_submodels()
        self.set_convection_submodel()
        self.set_intercalation_kinetics_submodel()
        self.set_other_reaction_submodels_to_zero()
        self.set_particle_submodel()
        self.set_solid_submodel()
        self.set_electrolyte_submodel()
        self.set_thermal_submodel()
        self.set_current_collector_submodel()

        self.set_sei_submodel()
        self.set_lithium_plating_submodel()

        if self.half_cell:
            # This also removes "negative electrode" submodels, so should be done last
            self.set_li_metal_counter_electrode_submodels()

        if build:
            self.build_model()

        if self.__class__ != "MPM":
            pybamm.citations.register("Marquis2019")

    def set_convection_submodel(self):

        self.submodels[
            "through-cell convection"
        ] = pybamm.convection.through_cell.NoConvection(self.param, self.options)
        self.submodels[
            "transverse convection"
        ] = pybamm.convection.transverse.NoConvection(self.param, self.options)

    def set_intercalation_kinetics_submodel(self):

        if self.options["surface form"] == "false":
            self.submodels["negative interface"] = self.inverse_intercalation_kinetics(
                self.param, "Negative", "lithium-ion main", self.options
            )
            self.submodels["positive interface"] = self.inverse_intercalation_kinetics(
                self.param, "Positive", "lithium-ion main", self.options
            )
            self.submodels[
                "negative interface current"
            ] = pybamm.kinetics.CurrentForInverseButlerVolmer(
                self.param, "Negative", "lithium-ion main", self.options
            )
            self.submodels[
                "positive interface current"
            ] = pybamm.kinetics.CurrentForInverseButlerVolmer(
                self.param, "Positive", "lithium-ion main", self.options
            )
        else:
            for domain in ["Negative", "Positive"]:
                intercalation_kinetics = self.get_intercalation_kinetics(domain)
                self.submodels[domain.lower() + " interface"] = intercalation_kinetics(
                    self.param, domain, "lithium-ion main", self.options
                )

    def set_particle_submodel(self):
        for domain in ["Negative", "Positive"]:
            particle = getattr(self.options, domain.lower())["particle"]
            if particle == "Fickian diffusion":
                self.submodels[
                    domain.lower() + " particle"
                ] = pybamm.particle.no_distribution.XAveragedFickianDiffusion(
                    self.param, domain, self.options
                )
            elif particle in [
                "uniform profile",
                "quadratic profile",
                "quartic profile",
            ]:
                self.submodels[
                    domain.lower() + " particle"
                ] = pybamm.particle.no_distribution.XAveragedPolynomialProfile(
                    self.param, domain, particle, self.options
                )

    def set_solid_submodel(self):

        self.submodels[
            "negative electrode potential"
        ] = pybamm.electrode.ohm.LeadingOrder(
            self.param, "Negative", options=self.options
        )
        self.submodels[
            "positive electrode potential"
        ] = pybamm.electrode.ohm.LeadingOrder(
            self.param, "Positive", options=self.options
        )

    def set_electrolyte_submodel(self):

        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        if self.options["electrolyte conductivity"] not in ["default", "leading order"]:
            raise pybamm.OptionError(
                "electrolyte conductivity '{}' not suitable for SPM".format(
                    self.options["electrolyte conductivity"]
                )
            )

        if self.options["surface form"] == "false" or self.half_cell:
            self.submodels[
                "leading-order electrolyte conductivity"
            ] = pybamm.electrolyte_conductivity.LeadingOrder(
                self.param, options=self.options
            )
        if self.options["surface form"] == "false":
            surf_model = surf_form.Explicit
        elif self.options["surface form"] == "differential":
            surf_model = surf_form.LeadingOrderDifferential
        elif self.options["surface form"] == "algebraic":
            surf_model = surf_form.LeadingOrderAlgebraic

        for domain in ["Negative", "Positive"]:
            self.submodels[
                domain.lower() + " surface potential difference"
            ] = surf_model(self.param, domain)

        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte_diffusion.ConstantConcentration(self.param, self.options)
