#
# Base class for particle cracking models.
#
import pybamm


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

    def __init__(self, param, domain, options, phase="primary"):
        if options["particle size"] == "distribution":
            self.size_distribution = True
        else:
            self.size_distribution = False
        super().__init__(param, domain, options=options, phase=phase)

    def _get_standard_variables(self, l_cr):
        domain, Domain = self.domain_Domain
        l_cr_av = pybamm.x_average(l_cr)
        variables = {
            f"{Domain} {self.phase_param.phase_name}particle crack length [m]": l_cr,
            f"X-averaged {domain} {self.phase_param.phase_name}particle crack length [m]": l_cr_av,
        }
        return variables

    def _get_standard_size_distribution_variables(self, l_cr_dist):
        domain, Domain = self.domain_Domain
        l_cr_av_dist = pybamm.x_average(l_cr_dist)
        variables = {
            f"{Domain} {self.phase_param.phase_name}particle crack length distribution [m]": l_cr_dist,
            f"X-averaged {domain} {self.phase_param.phase_name}particle crack length distribution [m]": l_cr_av_dist,
        }
        return variables

    def _get_mechanical_size_distribution_results(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name
        phase_param = self.phase_param
        c_s_rav = variables[
            f"R-averaged {domain} {phase_name}particle concentration distribution [mol.m-3]"
        ]
        c_s_surf = variables[
            f"{Domain} {phase_name}particle surface concentration distribution [mol.m-3]"
        ]
        T = pybamm.PrimaryBroadcast(
            variables[f"{Domain} electrode temperature [K]"],
            [f"{domain} {phase_name}particle size"],
        )
        T = pybamm.PrimaryBroadcast(
            T,
            [f"{domain} {phase_name}particle"],
        )

        # use a tangential approximation for omega
        c_0 = phase_param.c_0
        R0 = phase_param.R
        sto = variables[f"{Domain} {phase_name}particle concentration distribution"]
        Omega = pybamm.r_average(phase_param.Omega(sto, T))

        E0 = pybamm.r_average(phase_param.E(sto, T))
        nu = phase_param.nu
        # Ai2019 eq [10]
        disp_surf = Omega * R0 / 3 * (c_s_rav - c_0)
        # c0 reference concentration for no deformation
        # stress evaluated at the surface of the particles
        # Ai2019 eq [7] with r=R
        stress_r_surf = pybamm.Scalar(0)
        # Ai2019 eq [8] with r=R
        # c_s_rav is already multiplied by 3/R^3 inside r_average
        stress_t_surf = Omega * E0 * (c_s_rav - c_s_surf) / 3.0 / (1.0 - nu)

        # Averages
        stress_r_surf_av = pybamm.x_average(stress_r_surf)
        stress_t_surf_av = pybamm.x_average(stress_t_surf)
        disp_surf_av = pybamm.x_average(disp_surf)

        variables.update(
            {
                f"{Domain} {phase_name}particle surface radial stress distribution [Pa]": stress_r_surf,
                f"{Domain} {phase_name}particle surface tangential stress distribution [Pa]": stress_t_surf,
                f"{Domain} {phase_name}particle surface displacement distribution [m]": disp_surf,
                f"X-averaged {domain} {phase_name}particle surface "
                "radial stress distribution [Pa]": stress_r_surf_av,
                f"X-averaged {domain} {phase_name}particle surface "
                "tangential stress distribution [Pa]": stress_t_surf_av,
                f"X-averaged {domain} {phase_name}particle surface displacement distribution [m]": disp_surf_av,
            }
        )
        return variables

    def _get_mechanical_results(self, variables):
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

    def _extract_mechanical_variables(self, variables):
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

    def _compute_stress_and_displacement(self, mech_vars):
        """
        Compute particle surface stress and displacement.

        Based on Ai2019 equations:
        - Eq [7]: Radial stress at surface (r=R) is zero
        - Eq [8]: Tangential stress from concentration gradient
        - Eq [10]: Surface displacement from volume change
        """
        # Constants for stress calculations
        STRESS_GEOMETRIC_FACTOR = 3.0
        DISPLACEMENT_GEOMETRIC_FACTOR = 3.0

        # Surface displacement (Ai2019 eq [10])
        # Reference concentration c_0 gives zero deformation
        disp_surf = (
            mech_vars["Omega"]
            * mech_vars["R"]
            * (mech_vars["c_s_rav"] - mech_vars["c_0"])
            / DISPLACEMENT_GEOMETRIC_FACTOR
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
            / STRESS_GEOMETRIC_FACTOR
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

    def _add_particle_mechanics_variables(self, variables, stress_disp):
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

    def _aggregate_thickness_changes(self, variables, mech_vars):
        """
        Aggregate thickness changes from phases to electrode level and from
        electrodes to cell level.
        """
        domain, Domain = self.domain_Domain

        # Aggregate primary and secondary phase thickness changes
        self._aggregate_phase_thickness_changes(variables, Domain)

        # Aggregate negative and positive electrode thickness changes to cell level
        self._aggregate_cell_thickness_change(variables, mech_vars["T_xav"])

    def _aggregate_phase_thickness_changes(self, variables, Domain):
        """Combine primary and secondary phase thickness changes for an electrode."""
        primary_key = f"{Domain} electrode primary thickness change [m]"
        secondary_key = f"{Domain} electrode secondary thickness change [m]"
        combined_key = f"{Domain} electrode thickness change [m]"

        if primary_key in variables and secondary_key in variables:
            variables[combined_key] = variables[primary_key] + variables[secondary_key]

    def _aggregate_cell_thickness_change(self, variables, T_xav):
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

    def _get_standard_surface_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name

        l_cr = variables[f"{Domain} {phase_name}particle crack length [m]"]
        a = variables[
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
        ]
        rho_cr = self.phase_param.rho_cr
        w_cr = self.phase_param.w_cr
        roughness = 1 + 2 * l_cr * rho_cr * w_cr  # ratio of cracks to normal surface
        a_cr = (roughness - 1) * a  # crack surface area to volume ratio

        roughness_xavg = pybamm.x_average(roughness)
        variables = {
            f"{Domain} {phase_name}crack surface to volume ratio [m-1]": a_cr,
            f"{Domain} {phase_name}electrode roughness ratio": roughness,
            f"X-averaged {domain} {phase_name}electrode roughness ratio": roughness_xavg,
        }
        return variables
