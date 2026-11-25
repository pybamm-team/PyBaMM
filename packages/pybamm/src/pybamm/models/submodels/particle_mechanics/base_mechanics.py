#
# Base class for particle cracking models.
#
from __future__ import annotations

from typing import TYPE_CHECKING

import pybamm

if TYPE_CHECKING:
    from typing import Any


class BaseMechanics(pybamm.BaseSubModel):
    """
    Base class for particle mechanics models, referenced from :footcite:t:`Ai2019` and
    :footcite:t:`Deshpande2012`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : dict, optional
        Dictionary of either the electrode for "positive" or "Negative"
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")

    """

    # Physical constants for stress/displacement calculations (Ai2019)
    STRESS_GEOMETRIC_FACTOR = 3.0
    DISPLACEMENT_GEOMETRIC_FACTOR = 3.0
    CRACK_ROUGHNESS_FACTOR = 2.0

    def __init__(
        self,
        param: Any,
        domain: str | dict[str, str],
        options: dict[str, Any],
        phase: str = "primary",
    ) -> None:
        super().__init__(param, domain, options=options, phase=phase)
        self.size_distribution = (
            getattr(self.options, self.domain)["particle size"] == "distribution"
        )

    @property
    def _electrode_domains(self) -> dict[str, str]:
        return {"primary": f"{self.domain} electrode", "secondary": "current collector"}

    def _build_var_name(
        self, quantity: str, unit: str = "", averaged: bool = False
    ) -> str:
        """
        Construct standardized variable names for mechanics quantities.

        Parameters
        ----------
        quantity : str
            The physical quantity (e.g., "particle crack length")
        unit : str, optional
            The unit of measurement (e.g., "[m]")
        averaged : bool, optional
            Whether this is an x-averaged quantity

        Returns
        -------
        str
            The formatted variable name
        """
        domain, Domain = self.domain_Domain
        prefix = f"X-averaged {domain}" if averaged else Domain
        unit_str = f" {unit}" if unit else ""
        return f"{prefix} {self.phase_name}{quantity}{unit_str}"

    def _get_standard_variables(self, l_cr: pybamm.Symbol) -> dict[str, pybamm.Symbol]:
        """Generate standard crack length variables."""
        l_cr_av = pybamm.x_average(l_cr)
        return {
            self._build_var_name("particle crack length", "[m]"): l_cr,
            self._build_var_name(
                "particle crack length", "[m]", averaged=True
            ): l_cr_av,
        }

    def _get_standard_size_distribution_variables(
        self, l_cr_dist: pybamm.Symbol
    ) -> dict[str, pybamm.Symbol]:
        """Generate standard crack length distribution variables."""
        l_cr_av_dist = pybamm.x_average(l_cr_dist)
        return {
            self._build_var_name(
                "particle crack length distribution", "[m]"
            ): l_cr_dist,
            self._build_var_name(
                "particle crack length distribution", "[m]", averaged=True
            ): l_cr_av_dist,
        }

    def _get_zero_cracking_variables(self) -> dict[str, pybamm.Symbol]:
        """Crack length and cracking rate variables for particles that do not crack."""
        domain, Domain = self.domain_Domain
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0),
            f"{domain} electrode",
            {"secondary": "current collector"},
        )
        variables = self._get_standard_variables(zero)
        variables.update(
            {
                f"{Domain} particle cracking rate [m.s-1]": zero,
                f"X-averaged {domain} particle cracking rate [m.s-1]": pybamm.x_average(
                    zero
                ),
            }
        )
        return variables

    def _get_standard_surface_variables(
        self, l_cr: pybamm.Symbol
    ) -> dict[str, pybamm.Symbol]:
        """Calculate crack surface variables including roughness and surface area ratio."""
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        a = self.coupled_variable(
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]",
            domains=self._electrode_domains,
        )
        rho_cr = self.phase_param.rho_cr
        w_cr = self.phase_param.w_cr

        # Roughness is ratio of cracks to normal surface
        roughness = 1 + self.CRACK_ROUGHNESS_FACTOR * l_cr * rho_cr * w_cr
        # Crack surface area to volume ratio
        a_cr = (roughness - 1) * a

        roughness_xavg = pybamm.x_average(roughness)
        return {
            f"{Domain} {phase_name}crack surface to volume ratio [m-1]": a_cr,
            f"{Domain} {phase_name}electrode roughness ratio": roughness,
            f"X-averaged {domain} {phase_name}electrode roughness ratio": roughness_xavg,
        }

    def _get_mechanical_results(self) -> dict[str, pybamm.Symbol]:
        """
        Calculate particle surface stress and displacement and the electrode and cell
        thickness changes, following :footcite:t:`Ai2019`.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        phase_param = self.phase_param
        electrode_domains = self._electrode_domains

        c_s_rav = self.coupled_variable(
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]",
            domains=electrode_domains,
        )
        sto_rav = self.coupled_variable(
            f"R-averaged {domain} {phase_name}particle concentration",
            domains=electrode_domains,
        )
        c_s_surf = self.coupled_variable(
            f"{Domain} {phase_name}particle surface concentration [mol.m-3]",
            domains=electrode_domains,
        )
        sto = self.coupled_variable(
            f"{Domain} {phase_name}particle concentration",
            domains={
                "primary": f"{domain} {phase_name}particle",
                "secondary": f"{domain} electrode",
                "tertiary": "current collector",
            },
        )
        eps_s = self.coupled_variable(
            f"{Domain} electrode {phase_name}active material volume fraction",
            domains=electrode_domains,
        )
        T = pybamm.PrimaryBroadcast(
            self._electrode_temperature(), [f"{domain} {phase_name}particle"]
        )

        # Material properties (tangential approximation for Omega)
        Omega = pybamm.r_average(phase_param.Omega(sto, T))
        E = pybamm.r_average(phase_param.E(sto, T))
        stress_disp = self._compute_stress_and_displacement(c_s_rav, c_s_surf, Omega, E)

        # Volume change for the thickness calculation
        sto_init = pybamm.r_average(phase_param.c_init / phase_param.c_max)
        v_change = pybamm.x_average(
            eps_s * phase_param.t_change(sto_rav)
        ) - pybamm.x_average(eps_s * phase_param.t_change(sto_init))
        electrode_thickness_change = (
            self.param.n_electrodes_parallel * v_change * self.domain_param.L
        )

        variables = {
            f"{Domain} {phase_name}particle surface radial stress [Pa]": stress_disp[
                "stress_r_surf"
            ],
            f"{Domain} {phase_name}particle surface tangential stress [Pa]": stress_disp[
                "stress_t_surf"
            ],
            f"{Domain} {phase_name}particle surface displacement [m]": stress_disp[
                "disp_surf"
            ],
            f"X-averaged {domain} {phase_name}particle surface radial stress [Pa]": stress_disp[
                "stress_r_surf_av"
            ],
            f"X-averaged {domain} {phase_name}particle surface tangential stress [Pa]": stress_disp[
                "stress_t_surf_av"
            ],
            f"X-averaged {domain} {phase_name}particle surface displacement [m]": stress_disp[
                "disp_surf_av"
            ],
        }
        variables.update(
            self._get_thickness_change_variables(electrode_thickness_change)
        )
        return variables

    def _get_mechanical_size_distribution_results(self) -> dict[str, pybamm.Symbol]:
        """
        Calculate particle surface stress and displacement for particle size
        distributions, following :footcite:t:`Ai2019`.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        phase_param = self.phase_param
        size_domains = {
            "primary": f"{domain} {phase_name}particle size",
            "secondary": f"{domain} electrode",
            "tertiary": "current collector",
        }

        c_s_rav = self.coupled_variable(
            f"R-averaged {domain} {phase_name}"
            "particle concentration distribution [mol.m-3]",
            domains=size_domains,
        )
        c_s_surf = self.coupled_variable(
            f"{Domain} {phase_name}"
            "particle surface concentration distribution [mol.m-3]",
            domains=size_domains,
        )
        sto = self.coupled_variable(
            f"{Domain} {phase_name}particle concentration distribution",
            domains={
                "primary": f"{domain} {phase_name}particle",
                "secondary": f"{domain} {phase_name}particle size",
                "tertiary": f"{domain} electrode",
                "quaternary": "current collector",
            },
        )
        # Broadcast temperature to particle size and particle domains
        T = pybamm.PrimaryBroadcast(
            self._electrode_temperature(), [f"{domain} {phase_name}particle size"]
        )
        T = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle"])

        # Material properties (tangential approximation for Omega)
        Omega = pybamm.r_average(phase_param.Omega(sto, T))
        E = pybamm.r_average(phase_param.E(sto, T))
        stress_disp = self._compute_stress_and_displacement(c_s_rav, c_s_surf, Omega, E)

        return {
            f"{Domain} {phase_name}"
            "particle surface radial stress distribution [Pa]": stress_disp[
                "stress_r_surf"
            ],
            f"{Domain} {phase_name}"
            "particle surface tangential stress distribution [Pa]": stress_disp[
                "stress_t_surf"
            ],
            f"{Domain} {phase_name}"
            "particle surface displacement distribution [m]": stress_disp["disp_surf"],
            f"X-averaged {domain} {phase_name}"
            "particle surface radial stress distribution [Pa]": stress_disp[
                "stress_r_surf_av"
            ],
            f"X-averaged {domain} {phase_name}"
            "particle surface tangential stress distribution [Pa]": stress_disp[
                "stress_t_surf_av"
            ],
            f"X-averaged {domain} {phase_name}"
            "particle surface displacement distribution [m]": stress_disp[
                "disp_surf_av"
            ],
        }

    def _electrode_temperature(self) -> pybamm.CoupledVariable:
        return self.coupled_variable(
            f"{self.domain.capitalize()} electrode temperature [K]",
            domains=self._electrode_domains,
        )

    def _compute_stress_and_displacement(
        self,
        c_s_rav: pybamm.Symbol,
        c_s_surf: pybamm.Symbol,
        Omega: pybamm.Symbol,
        E: pybamm.Symbol,
    ) -> dict[str, pybamm.Symbol]:
        """
        Compute particle surface stress and displacement.

        Based on Ai2019 equations:
        - Eq [7]: Radial stress at surface (r=R) is zero
        - Eq [8]: Tangential stress from concentration gradient
        - Eq [10]: Surface displacement from volume change
        """
        phase_param = self.phase_param

        # Surface displacement (Ai2019 eq [10])
        # Reference concentration c_0 gives zero deformation
        disp_surf = (
            Omega
            * phase_param.R
            * (c_s_rav - phase_param.c_0)
            / self.DISPLACEMENT_GEOMETRIC_FACTOR
        )

        # Surface stresses (Ai2019 eqs [7-8])
        # Radial stress at particle surface is zero (eq [7] with r=R)
        stress_r_surf = pybamm.Scalar(0)

        # Tangential stress from concentration gradient (eq [8] with r=R)
        # Note: c_s_rav is already multiplied by 3/R^3 inside r_average
        stress_t_surf = (
            Omega
            * E
            * (c_s_rav - c_s_surf)
            / self.STRESS_GEOMETRIC_FACTOR
            / (1.0 - phase_param.nu)
        )

        return {
            "stress_r_surf": stress_r_surf,
            "stress_t_surf": stress_t_surf,
            "disp_surf": disp_surf,
            "stress_r_surf_av": pybamm.x_average(stress_r_surf),
            "stress_t_surf_av": pybamm.x_average(stress_t_surf),
            "disp_surf_av": pybamm.x_average(disp_surf),
        }

    def _electrode_swells(self, domain: str) -> bool:
        """Whether every particle phase of a porous electrode swells."""
        if self.options.electrode_types[domain] != "porous":
            return False
        domain_options = getattr(self.options, domain)
        return all(
            getattr(domain_options, phase)["particle mechanics"] != "none"
            for phase in self.options.phases[domain]
        )

    def _get_thickness_change_variables(
        self, electrode_thickness_change: pybamm.Symbol
    ) -> dict[str, pybamm.Symbol]:
        """
        Thickness change of this phase and, from the last phase of an electrode, the
        total thickness change of the electrode and of the cell.
        """
        domain, Domain = self.domain_Domain
        variables = {
            f"{Domain} electrode {self.phase_name}thickness change [m]": (
                electrode_thickness_change
            )
        }
        phases = self.options.phases[domain]
        if self.phase != phases[-1] or not self._electrode_swells(domain):
            return variables

        total_thickness_change = electrode_thickness_change
        if len(phases) > 1:
            primary_thickness_change = self.coupled_variable(
                f"{Domain} electrode primary thickness change [m]",
                domain="current collector",
            )
            total_thickness_change = (
                primary_thickness_change + electrode_thickness_change
            )
            variables[f"{Domain} electrode thickness change [m]"] = (
                total_thickness_change
            )

        if domain == "positive" and self._electrode_swells("negative"):
            negative_thickness_change = self.coupled_variable(
                "Negative electrode thickness change [m]", domain="current collector"
            )
            T_xav = self.coupled_variable(
                "X-averaged cell temperature [K]", domain="current collector"
            )
            # Thermal expansion contribution (Ai2019 eq [13])
            thermal_expansion = self.param.alpha_T_cell * (T_xav - self.param.T_ref)
            variables["Cell thickness change [m]"] = (
                negative_thickness_change + total_thickness_change + thermal_expansion
            )
        return variables
