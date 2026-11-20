"""Dual-continuum particle submodels, and the names of their parameters."""

from __future__ import annotations

import pybamm
from pybamm_model_zoo.dual_continuum.isolated_sphere import isolated_sphere_modes


def parameter_name(domain: str, name: str, phase: str | None = None) -> str:
    """Full name of a dual-continuum parameter.

    Parameters
    ----------
    domain : str
        "Positive" or "Negative".
    name : str
        The parameter, e.g. "s0 surface average".
    phase : str, optional
        "primary" or "secondary", only for an electrode with several particle
        phases.

    Returns
    -------
    str
        "{Domain} electrode {name}", prefixed with "Primary: " or "Secondary: "
        when ``phase`` is given, as for PyBaMM's own particle parameters.

    Raises
    ------
    :class:`pybamm.DomainError`
        If ``domain`` is not "Positive" or "Negative".
    :class:`pybamm.OptionError`
        If ``phase`` is not "primary", "secondary" or None.
    """
    Domain = domain.capitalize()
    if Domain not in ("Positive", "Negative"):
        raise pybamm.DomainError(
            f"domain must be 'Positive' or 'Negative', not {domain}"
        )
    if phase not in (None, "primary", "secondary"):
        raise pybamm.OptionError(
            f"phase must be 'primary', 'secondary' or None, not {phase}"
        )
    prefix = f"{phase.capitalize()}: " if phase else ""
    return f"{prefix}{Domain} electrode {name}"


def time_constant_name(m: int, dimensionless: bool) -> str:
    """Name of the time constant of mode ``m``, without electrode and phase.

    Parameters
    ----------
    m : int
        Mode number, from 1.
    dimensionless : bool
        Whether the time constant is dimensionless.

    Returns
    -------
    str
        The name to pass to :func:`parameter_name`.
    """
    unit = "dimensionless" if dimensionless else "[s]"
    return f"transient closure time constant {m} {unit}"


class DCParticle(pybamm.particle.BaseParticle):
    """Homogenised active-material mass balance of the dual-continuum model.

    The volume-averaged concentration obeys
    :math:`\\partial_t c_{vol} = -a j / (\\varepsilon_s F)`, and the surface
    concentration is :math:`c_{vol}` (DC0), :math:`c_{vol} + \\langle s \\rangle_A j`
    (DC1) or :math:`c_{vol} + \\gamma j` (Yang), with ``j`` replaced by its
    electrode average when computed a priori. There is no radial profile:
    radial outputs are the volume average broadcast in r.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel.
    domain : str
        Either "negative" or "positive".
    options : dict
        PyBaMM model options.
    dc_options : :class:`DCModelOptions`
        Dual-continuum options.
    phase : str, optional
        Phase of the particle (default is "primary").
    """

    def __init__(self, param, domain, options, dc_options, phase="primary"):
        super().__init__(param, domain, options, phase)
        self.dc_options = dc_options

    def build(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        c_max = self.phase_param.c_max
        electrode_domains = self._domains()

        j = self._coupled_interfacial_current_density()
        a = self.coupled_variable(
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]",
            domains=electrode_domains,
        )
        eps_s = self.coupled_variable(
            f"{Domain} electrode {phase_name}active material volume fraction",
            domains=electrode_domains,
        )
        T = self._broadcast_to_particle(self._coupled_temperature())
        current = self.coupled_variable("Total current density [A.m-2]")

        c_vol = pybamm.Variable(
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]",
            domains=electrode_domains,
            bounds=(0, c_max),
            scale=c_max,
        )
        c_vol.print_name = f"c_vol_{domain[0]}"
        c_init = pybamm.r_average(self.phase_param.c_init)
        self.rhs = {c_vol: -a * j / (eps_s * self.param.F)}
        self.initial_conditions = {c_vol: c_init}
        self._build_closure_states(c_vol, j)

        if self.dc_options.implicit_surface_concentration:
            c_surf = pybamm.Variable(
                f"{Domain} {phase_name}particle surface concentration [mol.m-3]",
                domains=electrode_domains,
                bounds=(0, c_max),
                scale=c_max,
            )
            c_surf.print_name = f"c_surf_{domain[0]}"
            correction = self._surface_correction(c_vol, j)
            self.algebraic = {c_surf: (c_surf - c_vol - correction) / c_max}
            # Initial guess for the algebraic solver
            self.initial_conditions[c_surf] = c_init
        elif self.dc_options["model type"] == "DC0":
            c_surf = c_vol
        else:
            j_av = self._electrode_averaged_current_density()
            c_surf = c_vol + self._surface_correction(c_vol, j_av)

        variables = self._get_concentration_variables(c_vol, c_surf)
        c_s = variables[f"{Domain} {phase_name}particle concentration [mol.m-3]"]
        D_eff = self._get_effective_diffusivity(c_s, T, current)
        N_s = pybamm.FullBroadcastToEdges(
            0,
            [f"{domain} {phase_name}particle"],
            auxiliary_domains={
                "secondary": f"{domain} electrode",
                "tertiary": "current collector",
            },
        )
        variables.update(self._get_standard_diffusivity_variables(D_eff))
        variables.update(self._get_standard_flux_variables(N_s))
        variables[f"{Domain} electrode {phase_name}DC correction term [mol.m-3]"] = (
            c_surf - c_vol
        )
        if self.dc_options["model type"] in ("DC1", "DC1t"):
            variables[
                f"{Domain} electrode {phase_name}closure variable [mol.m-1.A-1]"
            ] = self._closure_variable(c_vol)
        self.variables.update(variables)

    def _build_closure_states(self, c_vol, j):
        """Add the states of a closure with memory; DC0, DC1 and Yang have none."""

    def _get_concentration_variables(self, c_vol, c_surf):
        """Standard concentration variables, uniform in r."""
        c_s = pybamm.PrimaryBroadcast(
            c_vol, [f"{self.domain} {self.phase_name}particle"]
        )
        return self._get_standard_concentration_variables(
            c_s, c_s_rav=c_vol, c_s_surf=c_surf
        )

    def _surface_correction(self, c_vol, j):
        """c_surf - c_vol for the reaction rate ``j`` [mol.m-3]."""
        if self.dc_options["model type"] == "Yang":
            return self._yang_coefficient(c_vol) * j
        return self._closure_variable(c_vol) * j

    def _electrode_averaged_current_density(self):
        """Average interfacial current density imposed by the applied current."""
        domain = self.domain
        a_av = self.coupled_variable(
            f"X-averaged {domain} electrode {self.phase_name}"
            "surface area to volume ratio [m-1]",
            domain="current collector",
        )
        i_cell = self.coupled_variable(
            "Current collector current density [A.m-2]", domain="current collector"
        )
        L = self.domain_param.L
        # Discharge (i_cell > 0) delithiates the negative, lithiates the positive
        sign = 1 if domain == "negative" else -1
        j_av = sign * i_cell / (a_av * L)
        return pybamm.PrimaryBroadcast(j_av, f"{domain} electrode")

    def _diffusivity(self, c_vol):
        """Particle diffusivity [m2.s-1] at the volume-averaged concentration."""
        return self.phase_param.D(c_vol, self._coupled_temperature())

    def _effective_radius(self):
        """Effective particle radius [m] set by the active material submodel."""
        return self.coupled_variable(
            f"{self.domain.capitalize()} electrode {self.phase_name}"
            "effective particle radius [m]",
            domains=self._domains(),
        )

    def _parameter(self, name):
        """DC parameter of this electrode and phase (see ``parameter_name``)."""
        # PyBaMM prefixes the phase only when the electrode has several phases
        phase = self.phase if self.phase_param.phase_prefactor else None
        return pybamm.Parameter(parameter_name(self.domain, name, phase))

    def _closure_variable(self, c_vol):
        """DC1 closure variable <s>_A [mol.m-1.A-1]."""
        F = self.param.F
        if self.dc_options["closure variable"] == "isolated sphere":
            # Analytical solution of the closure problem for an isolated sphere
            return -self._effective_radius() / (5 * self._diffusivity(c_vol) * F)
        if self.dc_options["closure variable"] == "dimensionless":
            s_star = self._parameter("s0 surface average dimensionless")
            return s_star * self.domain_param.L / (self._diffusivity(c_vol) * F)
        return self._parameter("s0 surface average")

    def _yang_coefficient(self, c_vol):
        """Yang and Tartakovsky's (2025) gamma [mol.m-1.A-1] (Paten et al., Eqs. 48-50)."""
        D = self._diffusivity(c_vol)
        l_k = self.phase_param.R
        a_d = self._parameter("Yang fitting parameter")

        t_d = l_k**2 / D
        # Offset avoids the sqrt singularity at t = 0
        l_d_growing = a_d * (D * (pybamm.t + 1e-200)) ** 0.5
        growing = pybamm.t < t_d / a_d**2
        l_d = l_d_growing * growing + l_k * (1 - growing)

        return (l_d**2 / (6 * l_k**2) - l_d / (2 * l_k)) * l_k / (D * self.param.F)


class TransientDCParticle(DCParticle):
    """Dual-continuum particle with the transient closure (DC1t).

    With :math:`\\langle s^t \\rangle_A = s_0 \\sum_m w_m (1 - e^{-t/\\theta_m})`,
    the convolution :math:`\\langle s^t \\rangle_A \\star \\partial_t j` is
    :math:`s_0 \\sum_m w_m y_m` with :math:`dy_m/dt = (j - y_m)/\\theta_m` and
    :math:`y_m(0) = 0`, for any history of ``j``. The weights are normalised
    by their sum, so the long-time limit is DC1.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel.
    domain : str
        Either "negative" or "positive".
    options : dict
        PyBaMM model options.
    dc_options : :class:`DCModelOptions`
        Dual-continuum options.
    phase : str, optional
        Phase of the particle (default is "primary").
    """

    @property
    def n_modes(self) -> int:
        """Number of exponential modes."""
        return self.dc_options["transient closure modes"]

    def _mode_name(self, m):
        Domain = self.domain.capitalize()
        return f"{Domain} electrode {self.phase_name}transient closure mode {m} [A.m-2]"

    def _build_closure_states(self, c_vol, j):
        """Add the modes y_m, driven by the reaction rate."""
        if self.dc_options["calculate surface concentration a priori"] == "true":
            j = self._electrode_averaged_current_density()
        self._mode_variables = []
        for m, (_, theta) in enumerate(self._modes(c_vol), start=1):
            mode = pybamm.Variable(
                self._mode_name(m),
                domain=f"{self.domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            mode.print_name = f"y_{self.domain[0]}{m}"
            self._mode_variables.append(mode)
            self.variables[self._mode_name(m)] = mode
            self.rhs[mode] = (j - mode) / theta
            self.initial_conditions[mode] = pybamm.Scalar(0)

    def _surface_correction(self, c_vol, j):
        """s0 sum_m w_m y_m, with normalised weights [mol.m-3]."""
        modes = self._modes(c_vol)
        total = sum(weight for weight, _ in modes)
        correction = sum(
            weight * mode
            for (weight, _), mode in zip(modes, self._mode_variables, strict=True)
        )
        return self._closure_variable(c_vol) * correction / total

    def _modes(self, c_vol):
        """Weight and time constant [s] of each mode."""
        closure = self.dc_options["closure variable"]
        if closure == "isolated sphere":
            scale = self._effective_radius() ** 2 / self._diffusivity(c_vol)
            weights, thetas = isolated_sphere_modes(self.n_modes)
            return [
                (pybamm.Scalar(w), theta * scale)
                for w, theta in zip(weights, thetas, strict=True)
            ]
        dimensionless = closure == "dimensionless"
        scale = (
            self.domain_param.L**2 / self._diffusivity(c_vol) if dimensionless else 1
        )
        return [
            (
                self._parameter(f"transient closure weight {m}"),
                self._parameter(time_constant_name(m, dimensionless)) * scale,
            )
            for m in range(1, self.n_modes + 1)
        ]
