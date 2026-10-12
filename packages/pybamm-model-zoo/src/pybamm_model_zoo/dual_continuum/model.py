"""Dual-continuum (DC) model built on PyBaMM's Doyle-Fuller-Newman model."""

from __future__ import annotations

import pybamm
import pybamm_model_zoo
from pybamm_model_zoo.dual_continuum.dc_active_material import (
    ConstantFromImage,
    ConstantSpherical,
)
from pybamm_model_zoo.dual_continuum.dc_options import DCModelOptions
from pybamm_model_zoo.dual_continuum.dc_particle import (
    DCParticle,
    TransientDCParticle,
)
from pybamm_model_zoo.dual_continuum.transient_closure import (
    isolated_sphere_modes,
    time_constant_name,
)

SLUG = "dual_continuum"


class DualContinuum(pybamm.lithium_ion.DFN):
    """Dual-continuum model of a lithium-ion cell (Paten et al., 2026).

    The DFN with its particle problem replaced by a homogenised mass balance
    for the volume-averaged active-material concentration, and a DC0, DC1,
    DC1t or Yang surface concentration. For a parameter set written for the
    DFN, use ``dc_options={"closure variable": "isolated sphere", "surface
    area": "spherical"}``.

    Parameters
    ----------
    options : dict, optional
        PyBaMM model options, as for :class:`pybamm.lithium_ion.DFN`. Use
        ``{"working electrode": "positive"}`` for a cathode half cell.
    dc_options : dict, optional
        Dual-continuum options, see :class:`DCModelOptions`.
    name : str, optional
        The model name.
    build : bool, optional
        Whether to build the model on instantiation.

    Examples
    --------
    >>> import pybamm_model_zoo as zoo
    >>> model = zoo.load("DualContinuum")(dc_options={"model type": "DC1"})
    """

    def __init__(
        self,
        options: dict | None = None,
        dc_options: dict | None = None,
        name: str = "Dual-continuum model",
        build: bool = True,
    ) -> None:
        self.dc_options = DCModelOptions(dc_options)
        super().__init__(options=options, name=name, build=build)
        pybamm_model_zoo.register_citation(SLUG, "PyBaMMModelZoo2026", "Paten2026")
        if self.dc_options["model type"] == "DC1t":
            pybamm_model_zoo.register_citation(SLUG, "Paten2026b")

    def set_particle_submodel(self):
        if (
            self.dc_options["calculate surface concentration a priori"] == "true"
            and self.options["operating mode"] != "current"
        ):
            raise pybamm.OptionError(
                "'calculate surface concentration a priori' requires galvanostatic "
                "operation ('operating mode': 'current'), since it uses the applied "
                "current to estimate the reaction rate"
            )
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            phases = self.options.phases[domain]
            if len(phases) > 1:
                raise pybamm.OptionError(
                    "The dual-continuum model supports a single particle phase"
                )
            particle = (
                TransientDCParticle
                if self.dc_options["model type"] == "DC1t"
                else DCParticle
            )
            for phase in phases:
                self.submodels[f"{domain} {phase} particle"] = particle(
                    self.param, domain, self.options, self.dc_options, phase
                )
                self.submodels[f"{domain} {phase} total particle concentration"] = (
                    pybamm.particle.TotalConcentration(
                        self.param, domain, self.options, phase
                    )
                )

    def _rebuild_param(self):
        dc_options = getattr(self, "dc_options", None)
        if dc_options is None or dc_options["lithium foil surface porosity"] == "false":
            super()._rebuild_param()
            return
        if self.options["working electrode"] == "both":
            raise pybamm.OptionError(
                "'lithium foil surface porosity' only applies to half cells"
            )
        self.param = LithiumIonParametersWithFoilPorosity(self.options)

    def set_active_material_submodel(self):
        super().set_active_material_submodel()
        if self.dc_options["surface area"] == "from image":
            submodel = ConstantFromImage
        else:
            submodel = ConstantSpherical
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            if getattr(self.options, domain)["loss of active material"] != "none":
                raise pybamm.OptionError(
                    "The dual-continuum model does not support loss of active material"
                )
            for phase in self.options.phases[domain]:
                self.submodels[f"{domain} {phase} active material"] = submodel(
                    self.param, domain, self.options, phase
                )

    @property
    def default_parameter_values(self) -> pybamm.ParameterValues:
        """The DFN's default parameter set plus the DC parameters.

        Image-based surface areas default to the DFN geometry (3 eps_s / R,
        no AM-CBD or AM-separator contact), and the closure variable to the
        isolated-sphere value -R / (5 D F) at the reference state, so the
        defaults describe the same cell as the DFN defaults. For DC1t, the
        modes are those fitted to the isolated sphere's transient closure
        variable, with time constants scaled by R^2 / D at the reference state.
        """
        values = super().default_parameter_values
        param = pybamm.LithiumIonParameters()
        values.update(
            {
                "CBD surface porosity": 0.5,
                "Separator surface porosity": values["Separator porosity"],
            },
            check_already_exists=False,
        )
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            Domain = domain.capitalize()
            domain_param = param.domain_params[domain]
            phase_param = domain_param.prim
            D_ref = phase_param.D(0.5 * phase_param.c_max, param.T_ref)
            s_sphere = values.evaluate(-phase_param.R_typ / (5 * D_ref * param.F))
            area = f"{Domain} electrode specific surface area from image"
            values.update(
                {
                    f"{Domain} electrode s0 surface average": float(s_sphere),
                    f"{Domain} electrode s0 surface average dimensionless": float(
                        s_sphere * values.evaluate(D_ref * param.F / domain_param.L)
                    ),
                    f"{area} (AM-electrolyte) [m-1]": float(
                        values.evaluate(phase_param.a_typ)
                    ),
                    f"{area} (AM-CBD) [m-1]": 0.0,
                    f"{area} (AM-separator) [m-1]": 0.0,
                    f"{Domain} electrode Yang fitting parameter": 1.0,
                },
                check_already_exists=False,
            )
            if self.dc_options["model type"] == "DC1t":
                values.update(
                    self._isolated_sphere_mode_values(
                        values, Domain, domain_param, D_ref
                    ),
                    check_already_exists=False,
                )
        return values

    def _isolated_sphere_mode_values(self, values, Domain, domain_param, D_ref):
        """Isolated-sphere modes as DC1t parameters, dimensional and dimensionless."""
        R = domain_param.prim.R_typ
        diffusion_time = float(values.evaluate(R**2 / D_ref))
        thickness_time = float(values.evaluate(domain_param.L**2 / D_ref))
        prefix = f"{Domain} electrode"
        weights, time_constants = isolated_sphere_modes(
            self.dc_options["transient closure modes"]
        )
        mode_values = {}
        for m, (weight, theta) in enumerate(
            zip(weights, time_constants, strict=True), start=1
        ):
            mode_values[f"{prefix} transient closure weight {m}"] = weight
            mode_values[time_constant_name(prefix, m, False)] = theta * diffusion_time
            mode_values[time_constant_name(prefix, m, True)] = (
                theta * diffusion_time / thickness_time
            )
        return mode_values


class LithiumIonParametersWithFoilPorosity(pybamm.LithiumIonParameters):
    """Lithium-ion parameters with the lithium-foil reaction on the separator.

    The foil reacts only where the separator's pores meet it, so its exchange
    current density is scaled by the separator surface porosity.
    """

    def j0_Li_metal(self, c_e, c_Li, T):
        """Exchange-current density of the lithium foil [A.m-2]."""
        porosity = pybamm.Parameter("Separator surface porosity")
        return porosity * super().j0_Li_metal(c_e, c_Li, T)
