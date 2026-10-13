"""Dual-continuum particle submodel."""

from __future__ import annotations

import pybamm
from pybamm_model_zoo.dual_continuum.transient_closure import (
    isolated_sphere_modes,
    time_constant_name,
)


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
        if self.size_distribution:
            raise pybamm.OptionError(
                "The dual-continuum particle does not support particle-size "
                "distributions"
            )
        self.dc_options = dc_options

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        c_max = self.phase_param.c_max

        c_vol = pybamm.Variable(
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]",
            domain=f"{domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, c_max),
            scale=c_max,
        )
        c_vol.print_name = f"c_vol_{domain[0]}"
        variables = {
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]": c_vol
        }

        if self.dc_options.implicit_surface_concentration:
            c_surf = pybamm.Variable(
                f"{Domain} {phase_name}particle surface concentration [mol.m-3]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, c_max),
                scale=c_max,
            )
            c_surf.print_name = f"c_surf_{domain[0]}"
            variables.update(self._get_concentration_variables(c_vol, c_surf))

        return variables

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_vol = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
        if not self.dc_options.implicit_surface_concentration:
            if self.dc_options["model type"] == "DC0":
                c_surf = c_vol
            else:
                j_av = self._electrode_averaged_current_density(variables)
                c_surf = c_vol + self._surface_correction(variables, j_av)
            variables.update(self._get_concentration_variables(c_vol, c_surf))

        c_s = variables[f"{Domain} {phase_name}particle concentration [mol.m-3]"]
        c_surf = variables[
            f"{Domain} {phase_name}particle surface concentration [mol.m-3]"
        ]
        T = pybamm.PrimaryBroadcast(
            variables[f"{Domain} electrode temperature [K]"],
            [f"{domain} {phase_name}particle"],
        )
        current = variables["Total current density [A.m-2]"]
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
            ] = self._closure_variable(variables)
        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_vol = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
        j = variables[
            f"{Domain} electrode {phase_name}interfacial current density [A.m-2]"
        ]
        a = variables[
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
        ]
        eps_s = variables[
            f"{Domain} electrode {phase_name}active material volume fraction"
        ]
        self.rhs = {c_vol: -a * j / (eps_s * self.param.F)}

    def set_algebraic(self, variables):
        if not self.dc_options.implicit_surface_concentration:
            return

        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_vol = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
        c_surf = variables[
            f"{Domain} {phase_name}particle surface concentration [mol.m-3]"
        ]
        j = variables[
            f"{Domain} electrode {phase_name}interfacial current density [A.m-2]"
        ]

        correction = self._surface_correction(variables, j)
        self.algebraic = {
            c_surf: (c_surf - c_vol - correction) / self.phase_param.c_max
        }

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_init = pybamm.r_average(self.phase_param.c_init)
        c_vol = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
        self.initial_conditions = {c_vol: c_init}

        if self.dc_options.implicit_surface_concentration:
            # Initial guess for the algebraic solver
            c_surf = variables[
                f"{Domain} {phase_name}particle surface concentration [mol.m-3]"
            ]
            self.initial_conditions[c_surf] = c_init

    def _get_concentration_variables(self, c_vol, c_surf):
        """Standard concentration variables, uniform in r."""
        c_s = pybamm.PrimaryBroadcast(
            c_vol, [f"{self.domain} {self.phase_name}particle"]
        )
        return self._get_standard_concentration_variables(
            c_s, c_s_rav=c_vol, c_s_surf=c_surf
        )

    def _surface_correction(self, variables, j):
        """c_surf - c_vol for the reaction rate ``j`` [mol.m-3]."""
        if self.dc_options["model type"] == "Yang":
            return self._yang_coefficient(variables) * j
        return self._closure_variable(variables) * j

    def _electrode_averaged_current_density(self, variables):
        """Average interfacial current density imposed by the applied current."""
        domain = self.domain
        a_av = variables[
            f"X-averaged {domain} electrode {self.phase_name}"
            "surface area to volume ratio [m-1]"
        ]
        i_cell = variables["Current collector current density [A.m-2]"]
        L = self.domain_param.L
        # Discharge (i_cell > 0) delithiates the negative, lithiates the positive
        sign = 1 if domain == "negative" else -1
        j_av = sign * i_cell / (a_av * L)
        return pybamm.PrimaryBroadcast(j_av, f"{domain} electrode")

    def _diffusivity(self, variables):
        domain, Domain = self.domain_Domain
        c_vol = variables[
            f"R-averaged {domain} {self.phase_name}particle concentration [mol.m-3]"
        ]
        T = variables[f"{Domain} electrode temperature [K]"]
        return self.phase_param.D(c_vol, T)

    def _closure_variable(self, variables):
        """DC1 closure variable <s>_A [mol.m-1.A-1]."""
        Domain = self.domain.capitalize()
        F = self.param.F
        if self.dc_options["closure variable"] == "isolated sphere":
            # Analytical solution of the closure problem for an isolated sphere
            R_eff = variables[
                f"{Domain} electrode {self.phase_name}effective particle radius [m]"
            ]
            return -R_eff / (5 * self._diffusivity(variables) * F)
        if self.dc_options["closure variable"] == "dimensionless":
            s_star = pybamm.Parameter(
                f"{Domain} electrode s0 surface average dimensionless"
            )
            return s_star * self.domain_param.L / (self._diffusivity(variables) * F)
        return pybamm.Parameter(f"{Domain} electrode s0 surface average")

    def _yang_coefficient(self, variables):
        """Yang and Tartakovsky's (2025) gamma [mol.m-1.A-1] (Paten et al., Eqs. 48-50)."""
        Domain = self.domain.capitalize()
        D = self._diffusivity(variables)
        l_k = self.phase_param.R
        a_d = pybamm.Parameter(f"{Domain} electrode Yang fitting parameter")

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

    def get_fundamental_variables(self):
        variables = super().get_fundamental_variables()
        for m in range(1, self.n_modes + 1):
            mode = pybamm.Variable(
                self._mode_name(m),
                domain=f"{self.domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            mode.print_name = f"y_{self.domain[0]}{m}"
            variables[self._mode_name(m)] = mode
        return variables

    def set_rhs(self, variables):
        super().set_rhs(variables)
        j = self._mode_driver(variables)
        for (_, theta), m in zip(
            self._modes(variables), range(1, self.n_modes + 1), strict=True
        ):
            mode = variables[self._mode_name(m)]
            self.rhs[mode] = (j - mode) / theta

    def set_initial_conditions(self, variables):
        super().set_initial_conditions(variables)
        for m in range(1, self.n_modes + 1):
            self.initial_conditions[variables[self._mode_name(m)]] = pybamm.Scalar(0)

    def _surface_correction(self, variables, j):
        """s0 sum_m w_m y_m, with normalised weights [mol.m-3]."""
        modes = self._modes(variables)
        total = sum(weight for weight, _ in modes)
        correction = sum(
            weight * variables[self._mode_name(m)]
            for (weight, _), m in zip(modes, range(1, self.n_modes + 1), strict=True)
        )
        return self._closure_variable(variables) * correction / total

    def _mode_driver(self, variables):
        """Reaction rate that drives the modes [A.m-2]."""
        if self.dc_options["calculate surface concentration a priori"] == "true":
            return self._electrode_averaged_current_density(variables)
        Domain = self.domain.capitalize()
        return variables[
            f"{Domain} electrode {self.phase_name}interfacial current density [A.m-2]"
        ]

    def _modes(self, variables):
        """Weight and time constant [s] of each mode."""
        Domain = self.domain.capitalize()
        prefix = f"{Domain} electrode"
        closure = self.dc_options["closure variable"]
        if closure == "isolated sphere":
            radius = variables[
                f"{prefix} {self.phase_name}effective particle radius [m]"
            ]
            scale = radius**2 / self._diffusivity(variables)
            weights, thetas = isolated_sphere_modes(self.n_modes)
            return [
                (pybamm.Scalar(w), theta * scale)
                for w, theta in zip(weights, thetas, strict=True)
            ]
        dimensionless = closure == "dimensionless"
        scale = (
            self.domain_param.L**2 / self._diffusivity(variables)
            if dimensionless
            else 1
        )
        return [
            (
                pybamm.Parameter(f"{prefix} transient closure weight {m}"),
                pybamm.Parameter(time_constant_name(prefix, m, dimensionless)) * scale,
            )
            for m in range(1, self.n_modes + 1)
        ]
