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
    parameter_name,
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
        elif self.dc_options["model type"] == "Yang":
            pybamm_model_zoo.register_citation(SLUG, "Yang2025")

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
            if (
                len(phases) > 1
                and self.dc_options["calculate surface concentration a priori"]
                == "true"
            ):
                raise pybamm.OptionError(
                    "'calculate surface concentration a priori' requires a single "
                    "particle phase: how the current splits between phases is not "
                    "known a priori"
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

        See :meth:`add_dc_parameters`. PyBaMM's default set has no parameters
        for several particle phases, so for a composite electrode the DC
        parameters are left to :meth:`add_dc_parameters` with a suitable set.
        """
        values = super().default_parameter_values
        try:
            return self.add_dc_parameters(values)
        except KeyError:
            if not any(len(phases) > 1 for phases in self.options.phases.values()):
                raise
            return values

    def add_dc_parameters(
        self, parameter_values: pybamm.ParameterValues
    ) -> pybamm.ParameterValues:
        """A copy of ``parameter_values`` with the DC parameters it lacks.

        The added values describe the same cell as the DFN: image-based surface
        areas equal to 3 eps_s / R (no AM-CBD or AM-separator contact), the
        isolated-sphere closure variable -R / (5 D F) at the reference state
        and, for DC1t, the isolated sphere's modes with time constants scaled
        by R^2 / D. Each particle phase gets its own values.

        Parameters
        ----------
        parameter_values : :class:`pybamm.ParameterValues`
            A parameter set for the DFN with the same options.

        Returns
        -------
        :class:`pybamm.ParameterValues`
            The parameter set with the DC parameters added.
        """
        values = parameter_values.copy()
        param = self.param
        defaults = {
            "CBD surface porosity": 0.5,
            "Separator surface porosity": values["Separator porosity"],
        }
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            domain_param = param.domain_params[domain]
            phases = self.options.phases[domain]
            for phase in phases:
                phase_param = domain_param.phase_params[phase]
                defaults.update(
                    self._phase_defaults(
                        values, domain, phase if len(phases) > 1 else None, phase_param
                    )
                )
        values.update(
            {k: v for k, v in defaults.items() if k not in values},
            check_already_exists=False,
        )
        return values

    def _phase_defaults(self, values, domain, phase, phase_param):
        """DFN-equivalent DC parameters of one electrode and particle phase."""
        param = self.param
        L = param.domain_params[domain].L
        D_ref = phase_param.D(0.5 * phase_param.c_max, param.T_ref)
        R = phase_param.R_typ
        s_sphere = float(values.evaluate(-R / (5 * D_ref * param.F)))
        diffusion_time = float(values.evaluate(R**2 / D_ref))
        thickness_time = float(values.evaluate(L**2 / D_ref))

        def name(parameter):
            return parameter_name(domain, parameter, phase)

        area = "specific surface area from image"
        defaults = {
            name("s0 surface average"): s_sphere,
            name("s0 surface average dimensionless"): s_sphere
            * float(values.evaluate(D_ref * param.F / L)),
            name(f"{area} (AM-electrolyte) [m-1]"): float(
                values.evaluate(phase_param.a_typ)
            ),
            name(f"{area} (AM-CBD) [m-1]"): 0.0,
            name(f"{area} (AM-separator) [m-1]"): 0.0,
            name("Yang fitting parameter"): 1.0,
        }
        if self.dc_options["model type"] == "DC1t":
            weights, time_constants = isolated_sphere_modes(
                self.dc_options["transient closure modes"]
            )
            for m, (weight, theta) in enumerate(
                zip(weights, time_constants, strict=True), start=1
            ):
                defaults[name(f"transient closure weight {m}")] = weight
                defaults[name(time_constant_name(m, False))] = theta * diffusion_time
                defaults[name(time_constant_name(m, True))] = (
                    theta * diffusion_time / thickness_time
                )
        return defaults


class LithiumIonParametersWithFoilPorosity(pybamm.LithiumIonParameters):
    """Lithium-ion parameters with the lithium-foil reaction on the separator.

    The foil reacts only where the separator's pores meet it, so its exchange
    current density is scaled by the separator surface porosity.
    """

    def j0_Li_metal(self, c_e, c_Li, T):
        """Exchange-current density of the lithium foil [A.m-2]."""
        porosity = pybamm.Parameter("Separator surface porosity")
        return porosity * super().j0_Li_metal(c_e, c_Li, T)
