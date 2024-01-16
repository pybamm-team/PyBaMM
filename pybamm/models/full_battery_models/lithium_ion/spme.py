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
                self.submodels[
                    "electrolyte conductivity"
                ] = pybamm.electrolyte_conductivity.Composite(
                    self.param, options=self.options
                )
            elif self.options["electrolyte conductivity"] == "integrated":
                self.submodels[
                    "electrolyte conductivity"
                ] = pybamm.electrolyte_conductivity.Integrated(
                    self.param, options=self.options
                )
        if self.options["surface form"] == "false":
            surf_model = surf_form.Explicit
        elif self.options["surface form"] == "differential":
            surf_model = surf_form.CompositeDifferential
        elif self.options["surface form"] == "algebraic":
            surf_model = surf_form.CompositeAlgebraic

        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "porous":
                self.submodels[
                    f"{domain} surface potential difference [V]"
                ] = surf_model(self.param, domain, self.options)
