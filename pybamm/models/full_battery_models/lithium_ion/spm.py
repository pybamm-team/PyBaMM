#
# Single Particle Model (SPM)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class SPM(BaseModel):
    """
    Single Particle Model (SPM) of a lithium-ion battery, from
    :footcite:t:`Marquis2019`.
    See :class:`pybamm.lithium_ion.BaseModel` for more details.

    Examples
    --------
    >>> model = pybamm.lithium_ion.SPM()
    >>> model.name
    'Single Particle Model'

    """

    def __init__(self, options=None, name="Single Particle Model", build=True):
        # Use 'algebraic' surface form if non-default kinetics are used
        options = options or {}
        kinetics = options.get("intercalation kinetics")
        surface_form = options.get("surface form")
        if kinetics is not None and surface_form is None:
            options["surface form"] = "algebraic"

        # For degradation models we use the "x-average", note that for side reactions
        # this is set by "x-average side reactions"
        self.x_average = True

        # Set "x-average side reactions" to "true" if the model is SPM
        x_average_side_reactions = options.get("x-average side reactions")
        if x_average_side_reactions is None and self.__class__ in [
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.MPM,
        ]:
            options["x-average side reactions"] = "true"

        super().__init__(options, name)

        self.set_submodels(build)

        if self.__class__ != "MPM":
            pybamm.citations.register("Marquis2019")

        if (
            self.options["SEI"] not in ["none", "constant", ("constant", "none")]
            or self.options["lithium plating"] != "none"
        ):
            pybamm.citations.register("BrosaPlanella2022")

    def set_intercalation_kinetics_submodel(self):
        for domain in ["negative", "positive"]:
            electrode_type = self.options.electrode_types[domain]
            if electrode_type == "planar":
                continue

            if self.options["surface form"] == "false":
                inverse_intercalation_kinetics = (
                    self.get_inverse_intercalation_kinetics()
                )
                self.submodels[f"{domain} interface"] = inverse_intercalation_kinetics(
                    self.param, domain, "lithium-ion main", self.options
                )
                self.submodels[f"{domain} interface current"] = (
                    pybamm.kinetics.CurrentForInverseButlerVolmer(
                        self.param, domain, "lithium-ion main", self.options
                    )
                )
            else:
                intercalation_kinetics = self.get_intercalation_kinetics(domain)
                phases = self.options.phases[domain]
                for phase in phases:
                    submod = intercalation_kinetics(
                        self.param, domain, "lithium-ion main", self.options, phase
                    )
                    self.submodels[f"{domain} {phase} interface"] = submod
                if len(phases) > 1:
                    self.submodels[f"total {domain} interface"] = (
                        pybamm.kinetics.TotalMainKinetics(
                            self.param, domain, "lithium-ion main", self.options
                        )
                    )

    def set_particle_submodel(self):
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue

            particle = getattr(self.options, domain)["particle"]
            for phase in self.options.phases[domain]:
                if particle == "Fickian diffusion":
                    submod = pybamm.particle.FickianDiffusion(
                        self.param, domain, self.options, phase=phase, x_average=True
                    )
                elif particle in [
                    "uniform profile",
                    "quadratic profile",
                    "quartic profile",
                ]:
                    submod = pybamm.particle.XAveragedPolynomialProfile(
                        self.param, domain, self.options, phase=phase
                    )
                elif particle == "MSMR":
                    submod = pybamm.particle.MSMRDiffusion(
                        self.param, domain, self.options, phase=phase, x_average=True
                    )
                    # also set the submodel for calculating stoichiometry from
                    # potential
                    self.submodels[f"{domain} {phase} stoichiometry"] = (
                        pybamm.particle.MSMRStoichiometryVariables(
                            self.param,
                            domain,
                            self.options,
                            phase=phase,
                            x_average=True,
                        )
                    )
                self.submodels[f"{domain} {phase} particle"] = submod
                self.submodels[f"{domain} {phase} total particle concentration"] = (
                    pybamm.particle.TotalConcentration(
                        self.param, domain, self.options, phase
                    )
                )

    def set_solid_submodel(self):
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            self.submodels[f"{domain} electrode potential"] = (
                pybamm.electrode.ohm.LeadingOrder(self.param, domain, self.options)
            )

    def set_electrolyte_concentration_submodel(self):
        self.submodels["electrolyte diffusion"] = (
            pybamm.electrolyte_diffusion.ConstantConcentration(self.param, self.options)
        )

    def set_electrolyte_potential_submodel(self):
        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        if self.options["electrolyte conductivity"] not in ["default", "leading order"]:
            raise pybamm.OptionError(
                "electrolyte conductivity '{}' not suitable for SPM".format(
                    self.options["electrolyte conductivity"]
                )
            )

        if (
            self.options["surface form"] == "false"
            or self.options.electrode_types["negative"] == "planar"
        ):
            self.submodels["leading-order electrolyte conductivity"] = (
                pybamm.electrolyte_conductivity.LeadingOrder(
                    self.param, options=self.options
                )
            )
        if self.options["surface form"] == "false":
            surf_model = surf_form.Explicit
        elif self.options["surface form"] == "differential":
            surf_model = surf_form.LeadingOrderDifferential
        elif self.options["surface form"] == "algebraic":
            surf_model = surf_form.LeadingOrderAlgebraic

        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            self.submodels[f"{domain} surface potential difference"] = surf_model(
                self.param, domain, options=self.options
            )

    def set_summary_variables(self):
        self.set_default_summary_variables()
