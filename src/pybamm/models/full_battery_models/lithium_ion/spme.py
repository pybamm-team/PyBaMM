#
# Single Particle Model with Electrolyte (SPMe)
#
import pybamm

from .spm import SPM


class SPMe(SPM):
    """
    Single Particle Model with Electrolyte (SPMe) of a lithium-ion battery, from
    :footcite:t:`Marquis2019`. Inherits most submodels from SPM, only modifies
    potentials and electrolyte. See :class:`pybamm.lithium_ion.BaseModel` for more
    details.

    Examples
    --------
    >>> model = pybamm.lithium_ion.SPMe()
    >>> model.name
    'Single Particle Model with electrolyte'

    """

    def __init__(
        self, options=None, name="Single Particle Model with electrolyte", build=True
    ):
        # For degradation models we use the "x-average", note that for side reactions
        # this is overwritten by "x-average side reactions"
        self.x_average = True

        # Initialize with the SPM
        super().__init__(options, name, build)

    def set_solid_submodel(self):
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "porous":
                solid_submodel = pybamm.electrode.ohm.Composite
            elif self.options.electrode_types[domain] == "planar":
                if self.options["surface form"] == "false":
                    solid_submodel = pybamm.electrode.ohm.LithiumMetalExplicit
                else:
                    solid_submodel = pybamm.electrode.ohm.LithiumMetalSurfaceForm
            self.submodels[f"{domain} electrode potential"] = solid_submodel(
                self.param, domain, self.options
            )

    def set_electrolyte_concentration_submodel(self):
        self.submodels["electrolyte diffusion"] = pybamm.electrolyte_diffusion.Full(
            self.param, self.options
        )

    def set_electrolyte_potential_submodel(self):
        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        if (
            self.options["surface form"] == "false"
            or self.options.electrode_types["negative"] == "planar"
        ):
            if self.options["electrolyte conductivity"] in ["default", "composite"]:
                self.submodels["electrolyte conductivity"] = (
                    pybamm.electrolyte_conductivity.Composite(
                        self.param, options=self.options
                    )
                )
            elif self.options["electrolyte conductivity"] == "integrated":
                self.submodels["electrolyte conductivity"] = (
                    pybamm.electrolyte_conductivity.Integrated(
                        self.param, options=self.options
                    )
                )
        if self.options["surface form"] == "false":
            surf_model = surf_form.Explicit
        elif self.options["surface form"] == "differential":
            surf_model = surf_form.CompositeDifferential
        elif self.options["surface form"] == "algebraic":
            surf_model = surf_form.CompositeAlgebraic

        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "porous":
                self.submodels[f"{domain} surface potential difference [V]"] = (
                    surf_model(self.param, domain, self.options)
                )

        if self.options["surface form"] in ["false", "algebraic"]:
            for domain in ["negative", "positive"]:
                if self.options.electrode_types[domain] == "porous":
                    phi_s = self.variables.get(
                        f"{domain.capitalize()} electrode potential [V]"
                    )
                    phi_e = self.variables.get(
                        f"{domain.capitalize()} electrolyte potential [V]"
                    )
                    delta_phi = self.variables.get(
                        f"{domain} surface potential difference [V]"
                    )
                    delta_phi_av = self.variables.get(
                        f"X-averaged {domain} surface potential difference [V]"
                    )
                    if all([phi_s, phi_e, delta_phi, delta_phi_av]):
                        pybamm.logger.debug(
                            f"{domain.capitalize()} electrode for surface form "
                            f"'{self.options['surface form']}': "
                            f"phi_s={phi_s}, phi_e={phi_e}, delta_phi={delta_phi}, "
                            f"delta_phi_av={delta_phi_av}"
                        )
            voltage = self.variables.get("Terminal voltage [V]")
            if voltage:
                pybamm.logger.debug(
                    f"Terminal voltage for surface form '{self.options['surface form']}': {voltage}"
                )
            phases = self.options["particle phases"]
            if phases == "1" or phases == ("1", "1"):
                pybamm.logger.info(
                    "Surface forms 'false' and 'algebraic' should produce identical "
                    "results for single-phase configurations. If differences are observed, "
                    "check logs for phi_s, phi_e, delta_phi, and current densities."
                )
            else:
                pybamm.logger.warning(
                    "Surface form 'algebraic' is designed for multi-phase configurations. "
                    "Differences with 'false' may occur due to algebraic constraints in "
                    "CompositeAlgebraic. Verify with logs if unexpected."
                )
