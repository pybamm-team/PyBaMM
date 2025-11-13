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
        self.size_distribution = options["particle size"] == "distribution"
        super().__init__(param, domain, options=options, phase=phase)

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
        phase_name = self.phase_param.phase_name

        prefix = f"X-averaged {domain}" if averaged else Domain
        unit_str = f" {unit}" if unit else ""

        return f"{prefix} {phase_name}{quantity}{unit_str}"

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

    def _get_mechanical_size_distribution_results(
        self, variables: dict[str, pybamm.Symbol]
    ) -> dict[str, pybamm.Symbol]:
        """
        Calculate mechanical results for particle size distributions.

        Computes stress and displacement for particle size distributions,
        following the models in Ai2019 and Deshpande2012.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name
        phase_param = self.phase_param

        # Extract concentration variables
        c_s_rav = variables[
            f"R-averaged {domain} {phase_name}"
            "particle concentration distribution [mol.m-3]"
        ]
        c_s_surf = variables[
            f"{Domain} {phase_name}"
            "particle surface concentration distribution [mol.m-3]"
        ]
        sto = variables[f"{Domain} {phase_name}particle concentration distribution"]

        # Broadcast temperature to particle size and particle domains
        T = self._broadcast_temperature_to_particle_size(variables)

        # Compute material properties (tangential approximation for Omega)
        Omega = pybamm.r_average(phase_param.Omega(sto, T))
        E = pybamm.r_average(phase_param.E(sto, T))

        # Calculate stress and displacement (Ai2019 equations)
        stress_disp = self._compute_distribution_stress_displacement(
            c_s_rav, c_s_surf, Omega, E, phase_param
        )

        # Add results to variables dictionary
        self._add_distribution_mechanics_variables(variables, stress_disp)

        return variables

    def _broadcast_temperature_to_particle_size(
        self, variables: dict[str, pybamm.Symbol]
    ) -> pybamm.Symbol:
        """Broadcast electrode temperature to particle size and particle domains."""
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name

        T = pybamm.PrimaryBroadcast(
            variables[f"{Domain} electrode temperature [K]"],
            [f"{domain} {phase_name}particle size"],
        )
        return pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle"])

    def _compute_distribution_stress_displacement(
        self,
        c_s_rav: pybamm.Symbol,
        c_s_surf: pybamm.Symbol,
        Omega: pybamm.Symbol,
        E: pybamm.Symbol,
        phase_param: Any,
    ) -> dict[str, pybamm.Symbol]:
        """
        Compute stress and displacement for size distributions.

        Based on Ai2019 equations:
        - Eq [7]: Radial stress at surface (r=R) is zero
        - Eq [8]: Tangential stress from concentration gradient
        - Eq [10]: Surface displacement from volume change
        """
        c_0 = phase_param.c_0
        R0 = phase_param.R
        nu = phase_param.nu

        # Surface displacement (Ai2019 eq [10])
        # c_0 is reference concentration for no deformation
        disp_surf = Omega * R0 * (c_s_rav - c_0) / self.DISPLACEMENT_GEOMETRIC_FACTOR

        # Surface stresses (Ai2019 eqs [7-8])
        # Radial stress at particle surface is zero (eq [7] with r=R)
        stress_r_surf = pybamm.Scalar(0)

        # Tangential stress (eq [8] with r=R)
        # Note: c_s_rav is already multiplied by 3/R^3 inside r_average
        stress_t_surf = (
            Omega * E * (c_s_rav - c_s_surf) / self.STRESS_GEOMETRIC_FACTOR / (1.0 - nu)
        )

        return {
            "stress_r_surf": stress_r_surf,
            "stress_t_surf": stress_t_surf,
            "disp_surf": disp_surf,
            "stress_r_surf_av": pybamm.x_average(stress_r_surf),
            "stress_t_surf_av": pybamm.x_average(stress_t_surf),
            "disp_surf_av": pybamm.x_average(disp_surf),
        }

    def _add_distribution_mechanics_variables(
        self, variables: dict[str, pybamm.Symbol], stress_disp: dict[str, pybamm.Symbol]
    ) -> None:
        """Add distribution mechanics variables to the variables dictionary."""
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name

        variables.update(
            {
                f"{Domain} {phase_name}"
                "particle surface radial stress distribution [Pa]": stress_disp[
                    "stress_r_surf"
                ],
                f"{Domain} {phase_name}"
                "particle surface tangential stress distribution [Pa]": stress_disp[
                    "stress_t_surf"
                ],
                f"{Domain} {phase_name}"
                "particle surface displacement distribution [m]": stress_disp[
                    "disp_surf"
                ],
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
        )

    def _get_mechanical_results(
        self, variables: dict[str, pybamm.Symbol]
    ) -> dict[str, pybamm.Symbol]:
        """
        Calculate mechanical results including stress, displacement, and thickness changes.

        This method computes particle surface stress and displacement based on
        concentration gradients, following the mechanical models in Ai2019 and
        Deshpande2012.
        """
        # Extract required variables and compute particle mechanics
        mech_vars = self._extract_mechanical_variables(variables)
        stress_disp = self._compute_stress_and_displacement(mech_vars)

        # Update variables with computed results
        self._add_particle_mechanics_variables(variables, stress_disp)

        # Aggregate thickness changes at electrode and cell level
        self._aggregate_thickness_changes(variables, mech_vars)

        return variables

    def _extract_mechanical_variables(
        self, variables: dict[str, pybamm.Symbol]
    ) -> dict[str, Any]:
        """Extract and compute intermediate variables needed for mechanics calculations."""
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        phase_param = self.phase_param

        c_s_rav = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
        sto_rav = variables[f"R-averaged {domain} {phase_name}particle concentration"]
        c_s_surf = variables[
            f"{Domain} {phase_name}particle surface concentration [mol.m-3]"
        ]
        T_xav = variables["X-averaged cell temperature [K]"]

        # Broadcast temperature to particle domain
        T = pybamm.PrimaryBroadcast(
            variables[f"{Domain} electrode temperature [K]"],
            [f"{domain} {phase_name}particle"],
        )
        eps_s = variables[
            f"{Domain} electrode {phase_name}active material volume fraction"
        ]
        sto = variables[f"{Domain} {phase_name}particle concentration"]

        # Compute material properties (tangential approximation for Omega)
        Omega = pybamm.r_average(phase_param.Omega(sto, T))
        E = pybamm.r_average(phase_param.E(sto, T))
        sto_init = pybamm.r_average(phase_param.c_init / phase_param.c_max)

        # Compute volume change for thickness calculation
        v_change = pybamm.x_average(
            eps_s * phase_param.t_change(sto_rav)
        ) - pybamm.x_average(eps_s * phase_param.t_change(sto_init))

        electrode_thickness_change = (
            self.param.n_electrodes_parallel * v_change * self.domain_param.L
        )

        return {
            "c_s_rav": c_s_rav,
            "c_s_surf": c_s_surf,
            "c_0": phase_param.c_0,
            "R": phase_param.R,
            "Omega": Omega,
            "E": E,
            "nu": phase_param.nu,
            "T_xav": T_xav,
            "electrode_thickness_change": electrode_thickness_change,
        }

    def _compute_stress_and_displacement(
        self, mech_vars: dict[str, Any]
    ) -> dict[str, pybamm.Symbol]:
        """
        Compute particle surface stress and displacement.

        Based on Ai2019 equations:
        - Eq [7]: Radial stress at surface (r=R) is zero
        - Eq [8]: Tangential stress from concentration gradient
        - Eq [10]: Surface displacement from volume change
        """
        # Surface displacement (Ai2019 eq [10])
        # Reference concentration c_0 gives zero deformation
        disp_surf = (
            mech_vars["Omega"]
            * mech_vars["R"]
            * (mech_vars["c_s_rav"] - mech_vars["c_0"])
            / self.DISPLACEMENT_GEOMETRIC_FACTOR
        )

        # Surface stresses (Ai2019 eqs [7-8])
        # Radial stress at particle surface is zero (eq [7] with r=R)
        stress_r_surf = pybamm.Scalar(0)

        # Tangential stress from concentration gradient (eq [8] with r=R)
        # Note: c_s_rav is already multiplied by 3/R^3 inside r_average
        stress_t_surf = (
            mech_vars["Omega"]
            * mech_vars["E"]
            * (mech_vars["c_s_rav"] - mech_vars["c_s_surf"])
            / self.STRESS_GEOMETRIC_FACTOR
            / (1.0 - mech_vars["nu"])
        )

        return {
            "stress_r_surf": stress_r_surf,
            "stress_t_surf": stress_t_surf,
            "disp_surf": disp_surf,
            "stress_r_surf_av": pybamm.x_average(stress_r_surf),
            "stress_t_surf_av": pybamm.x_average(stress_t_surf),
            "disp_surf_av": pybamm.x_average(disp_surf),
            "electrode_thickness_change": mech_vars["electrode_thickness_change"],
        }

    def _add_particle_mechanics_variables(
        self,
        variables: dict[str, pybamm.Symbol],
        stress_disp: dict[str, pybamm.Symbol],
    ) -> None:
        """Add computed stress and displacement variables to the variables dictionary."""
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        variables.update(
            {
                # Spatial distributions
                f"{Domain} {phase_name}particle surface radial stress [Pa]": stress_disp[
                    "stress_r_surf"
                ],
                f"{Domain} {phase_name}particle surface tangential stress [Pa]": stress_disp[
                    "stress_t_surf"
                ],
                f"{Domain} {phase_name}particle surface displacement [m]": stress_disp[
                    "disp_surf"
                ],
                # X-averaged quantities
                f"X-averaged {domain} {phase_name}particle surface radial stress [Pa]": stress_disp[
                    "stress_r_surf_av"
                ],
                f"X-averaged {domain} {phase_name}particle surface tangential stress [Pa]": stress_disp[
                    "stress_t_surf_av"
                ],
                f"X-averaged {domain} {phase_name}particle surface displacement [m]": stress_disp[
                    "disp_surf_av"
                ],
                # Thickness change
                f"{Domain} electrode {phase_name}thickness change [m]": stress_disp[
                    "electrode_thickness_change"
                ],
            }
        )

    def _aggregate_thickness_changes(
        self, variables: dict[str, pybamm.Symbol], mech_vars: dict[str, Any]
    ) -> None:
        """
        Aggregate thickness changes from phases to electrode level and from
        electrodes to cell level.
        """
        domain, Domain = self.domain_Domain

        # Aggregate primary and secondary phase thickness changes
        self._aggregate_phase_thickness_changes(variables, Domain)

        # Aggregate negative and positive electrode thickness changes to cell level
        self._aggregate_cell_thickness_change(variables, mech_vars["T_xav"])

    def _aggregate_phase_thickness_changes(
        self, variables: dict[str, pybamm.Symbol], Domain: str
    ) -> None:
        """Combine primary and secondary phase thickness changes for an electrode."""
        primary_key = f"{Domain} electrode primary thickness change [m]"
        secondary_key = f"{Domain} electrode secondary thickness change [m]"
        combined_key = f"{Domain} electrode thickness change [m]"

        if primary_key in variables and secondary_key in variables:
            variables[combined_key] = variables[primary_key] + variables[secondary_key]

    def _aggregate_cell_thickness_change(
        self, variables: dict[str, pybamm.Symbol], T_xav: pybamm.Symbol
    ) -> None:
        """
        Calculate total cell thickness change from electrode changes and thermal expansion.

        Based on Ai2019 eq [13] for thermal expansion contribution.
        """
        neg_key = "Negative electrode thickness change [m]"
        pos_key = "Positive electrode thickness change [m]"

        if neg_key in variables and pos_key in variables:
            # Thermal expansion contribution (Ai2019 eq [13])
            thermal_expansion = self.param.alpha_T_cell * (T_xav - self.param.T_ref)

            # Total cell thickness change
            variables["Cell thickness change [m]"] = (
                variables[neg_key] + variables[pos_key] + thermal_expansion
            )

    def _get_standard_surface_variables(
        self, variables: dict[str, pybamm.Symbol]
    ) -> dict[str, pybamm.Symbol]:
        """Calculate crack surface variables including roughness and surface area ratio."""
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name

        l_cr = variables[f"{Domain} {phase_name}particle crack length [m]"]
        a = variables[
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
        ]
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
