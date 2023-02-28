#
# Base class for particles
#
import pybamm


class BaseParticle(pybamm.BaseSubModel):
    """
    Base class for molar conservation in particles.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options=options, phase=phase)
        # Read from options to see if we have a particle size distribution
        domain_options = getattr(self.options, domain)
        self.size_distribution = domain_options["particle size"] == "distribution"

    def _get_effective_diffusivity(self, c, T):
        param = self.param
        domain = self.domain
        domain_param = self.domain_param
        phase_param = self.phase_param

        # Get diffusivity
        D = phase_param.D(c, T)

        # Account for stress-induced difftusion by defining a multiplicative
        # "stress factor"
        stress_option = getattr(self.options, domain)["stress-induced diffusion"]

        if stress_option == "true":
            # Ai2019 eq [12]
            Omega = domain_param.Omega
            E = domain_param.E
            nu = domain_param.nu
            theta_M = Omega / (param.R * T) * (2 * Omega * E) / (9 * (1 - nu))
            stress_factor = 1 + theta_M * (c - domain_param.c_0)
        else:
            stress_factor = 1

        return D * stress_factor

    def _get_standard_concentration_variables(
        self, c_s, c_s_xav=None, c_s_rav=None, c_s_av=None, c_s_surf=None
    ):
        """
        All particle submodels must provide the particle concentration as an argument
        to this method. Some submodels solve for quantities other than the concentration
        itself, for example the 'XAveragedFickianDiffusion' models solves for the
        x-averaged concentration. In such cases the variables being solved for (set in
        'get_fundamental_variables') must also be passed as keyword arguments. If not
        passed as keyword arguments, the various average concentrations and surface
        concentration are computed automatically from the particle concentration.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        # Get surface concentration if not provided as fundamental variable to
        # solve for
        if c_s_surf is None:
            c_s_surf = pybamm.surf(c_s)
        c_s_surf_av = pybamm.x_average(c_s_surf)

        c_scale = self.phase_param.c_max

        # Get average concentration(s) if not provided as fundamental variable to
        # solve for
        if c_s_xav is None:
            c_s_xav = pybamm.x_average(c_s)
        if c_s_rav is None:
            c_s_rav = pybamm.r_average(c_s)
        if c_s_av is None:
            c_s_av = pybamm.r_average(c_s_xav)

        variables = {
            f"{Domain} {phase_name}particle stoichiometry": c_s / c_scale,
            f"{Domain} {phase_name}particle concentration": c_s / c_scale,
            f"{Domain} {phase_name}particle concentration [mol.m-3]": c_s,
            f"X-averaged {domain} {phase_name}particle concentration": c_s_xav
            / c_scale,
            f"X-averaged {domain} {phase_name}particle "
            "concentration [mol.m-3]": c_s_xav,
            f"R-averaged {domain} {phase_name}particle concentration": c_s_rav
            / c_scale,
            f"R-averaged {domain} {phase_name}particle "
            "concentration [mol.m-3]": c_s_rav,
            f"Average {domain} {phase_name}particle concentration": c_s_av / c_scale,
            f"Average {domain} {phase_name}particle concentration [mol.m-3]": c_s_av,
            f"{Domain} {phase_name}particle surface stoichiometry": c_s_surf / c_scale,
            f"{Domain} {phase_name}particle surface concentration": c_s_surf / c_scale,
            f"{Domain} {phase_name}particle surface concentration [mol.m-3]": c_s_surf,
            f"X-averaged {domain} {phase_name}particle "
            "surface concentration": c_s_surf_av / c_scale,
            f"X-averaged {domain} {phase_name}particle "
            "surface concentration [mol.m-3]": c_s_surf_av,
            f"{Domain} electrode extent of lithiation": c_s_rav / c_scale,
            f"X-averaged {domain} electrode extent of lithiation": c_s_av / c_scale,
            f"Minimum {domain} {phase_name}particle concentration": pybamm.min(c_s)
            / c_scale,
            f"Maximum {domain} {phase_name}particle concentration": pybamm.max(c_s)
            / c_scale,
            f"Minimum {domain} {phase_name}particle concentration [mol.m-3]"
            "": pybamm.min(c_s),
            f"Maximum {domain} {phase_name}particle concentration [mol.m-3]"
            "": pybamm.max(c_s),
            f"Minimum {domain} {phase_name}particle "
            "surface concentration": pybamm.min(c_s_surf) / c_scale,
            f"Maximum {domain} {phase_name}particle "
            "surface concentration": pybamm.max(c_s_surf) / c_scale,
            f"Minimum {domain} {phase_name}particle "
            "surface concentration [mol.m-3]": pybamm.min(c_s_surf),
            f"Maximum {domain} {phase_name}particle "
            "surface concentration [mol.m-3]": pybamm.max(c_s_surf),
        }

        return variables

    def _get_standard_flux_variables(self, N_s):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        variables = {f"{Domain} {phase_name}particle flux [mol.m-2.s-1]": N_s}

        if isinstance(N_s, pybamm.Broadcast):
            N_s_xav = pybamm.x_average(N_s)
            variables.update(
                {
                    f"X-averaged {domain} {phase_name}"
                    "particle flux [mol.m-2.s-1]": N_s_xav
                }
            )
        return variables

    def _get_distribution_variables(self, R):
        """
        Forms the particle-size distributions and mean radii given a spatial variable
        R. The domains of R will be different depending on the submodel, e.g. for the
        `SingleSizeDistribution` classes R does not have an "electrode" domain.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        R_typ = self.phase_param.R_typ  # [m]
        # Particle-size distribution (area-weighted)
        f_a_dist = self.phase_param.f_a_dist(R)  # [m-1]

        # Ensure the distribution is normalised, irrespective of discretisation
        # or user input
        f_a_dist = f_a_dist / pybamm.Integral(f_a_dist, R)  # [m-1]

        # Volume-weighted particle-size distribution
        f_v_dist = R * f_a_dist / pybamm.Integral(R * f_a_dist, R)  # [m-1]

        # Number-based particle-size distribution
        f_num_dist = (f_a_dist / R**2) / pybamm.Integral(
            f_a_dist / R**2, R
        )  # [m-1]

        # True mean radii and standard deviations, calculated from the f_a_dist that
        # was given, all have units [m]
        R_num_mean = pybamm.Integral(R * f_num_dist, R)
        R_a_mean = pybamm.Integral(R * f_a_dist, R)
        R_v_mean = pybamm.Integral(R * f_v_dist, R)
        sd_num = pybamm.sqrt(pybamm.Integral((R - R_num_mean) ** 2 * f_num_dist, R))
        sd_a = pybamm.sqrt(pybamm.Integral((R - R_a_mean) ** 2 * f_a_dist, R))
        sd_v = pybamm.sqrt(pybamm.Integral((R - R_v_mean) ** 2 * f_v_dist, R))

        # X-average the means and standard deviations to give scalars
        # (to remove the "electrode" domain, if present)
        R_num_mean = pybamm.x_average(R_num_mean)
        R_a_mean = pybamm.x_average(R_a_mean)
        R_v_mean = pybamm.x_average(R_v_mean)
        sd_num = pybamm.x_average(sd_num)
        sd_a = pybamm.x_average(sd_a)
        sd_v = pybamm.x_average(sd_v)

        # X-averaged distributions, or broadcast
        if R.domains["secondary"] == [f"{domain} electrode"]:
            f_a_dist_xav = pybamm.x_average(f_a_dist)
            f_v_dist_xav = pybamm.x_average(f_v_dist)
            f_num_dist_xav = pybamm.x_average(f_num_dist)
        else:
            f_a_dist_xav = f_a_dist
            f_v_dist_xav = f_v_dist
            f_num_dist_xav = f_num_dist

            # broadcast
            f_a_dist = pybamm.SecondaryBroadcast(f_a_dist_xav, [f"{domain} electrode"])
            f_v_dist = pybamm.SecondaryBroadcast(f_v_dist_xav, [f"{domain} electrode"])
            f_num_dist = pybamm.SecondaryBroadcast(
                f_num_dist_xav, [f"{domain} electrode"]
            )

        variables = {
            f"{Domain} {phase_name}particle sizes": R / R_typ,
            f"{Domain} {phase_name}particle sizes [m]": R,
            f"{Domain} area-weighted {phase_name}particle-size"
            " distribution [m-1]": f_a_dist,
            f"{Domain} volume-weighted {phase_name}particle-size"
            " distribution [m-1]": f_v_dist,
            f"{Domain} number-based {phase_name}particle-size"
            " distribution [m-1]": f_num_dist,
            f"{Domain} area-weighted mean particle radius [m]": R_a_mean,
            f"{Domain} volume-weighted mean particle radius [m]": R_v_mean,
            f"{Domain} number-based mean particle radius [m]": R_num_mean,
            f"{Domain} area-weighted {phase_name}particle-size"
            " standard deviation [m]": sd_a,
            f"{Domain} volume-weighted {phase_name}particle-size"
            " standard deviation [m]": sd_v,
            f"{Domain} number-based {phase_name}particle-size"
            " standard deviation [m]": sd_num,
            # X-averaged sizes and distributions
            f"X-averaged {domain} {phase_name}particle sizes [m]": pybamm.x_average(R),
            f"X-averaged {domain} area-weighted {phase_name}particle-size "
            "distribution [m-1]": f_a_dist_xav,
            f"X-averaged {domain} volume-weighted {phase_name}particle-size "
            "distribution [m-1]": f_v_dist_xav,
            f"X-averaged {domain} number-based {phase_name}particle-size "
            "distribution [m-1]": f_num_dist_xav,
        }

        return variables

    def _get_standard_concentration_distribution_variables(self, c_s):
        """
        Forms standard concentration variables that depend on particle size R given
        the fundamental concentration distribution variable c_s from the submodel.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_scale = self.phase_param.c_max
        # Broadcast and x-average when necessary
        if c_s.domain == [f"{domain} {phase_name}particle size"] and c_s.domains[
            "secondary"
        ] != [f"{domain} electrode"]:
            # X-avg concentration distribution
            c_s_xav_distribution = pybamm.PrimaryBroadcast(
                c_s, [f"{domain} {phase_name}particle"]
            )

            # Surface concentration distribution variables
            c_s_surf_xav_distribution = c_s
            c_s_surf_distribution = pybamm.SecondaryBroadcast(
                c_s_surf_xav_distribution, [f"{domain} electrode"]
            )

            # Concentration distribution in all domains.
            c_s_distribution = pybamm.PrimaryBroadcast(
                c_s_surf_distribution, [f"{domain} {phase_name}particle"]
            )
        elif c_s.domain == [f"{domain} {phase_name}particle"] and (
            c_s.domains["tertiary"] != [f"{domain} electrode"]
        ):
            # X-avg concentration distribution
            c_s_xav_distribution = c_s

            # Surface concentration distribution variables
            c_s_surf_xav_distribution = pybamm.surf(c_s_xav_distribution)
            c_s_surf_distribution = pybamm.SecondaryBroadcast(
                c_s_surf_xav_distribution, [f"{domain} electrode"]
            )

            # Concentration distribution in all domains.
            c_s_distribution = pybamm.TertiaryBroadcast(
                c_s_xav_distribution, [f"{domain} electrode"]
            )
        elif c_s.domain == [f"{domain} {phase_name}particle size"] and c_s.domains[
            "secondary"
        ] == [f"{domain} electrode"]:
            # Surface concentration distribution variables
            c_s_surf_distribution = c_s
            c_s_surf_xav_distribution = pybamm.x_average(c_s)

            # X-avg concentration distribution
            c_s_xav_distribution = pybamm.PrimaryBroadcast(
                c_s_surf_xav_distribution, [f"{domain} {phase_name}particle"]
            )

            # Concentration distribution in all domains.
            c_s_distribution = pybamm.PrimaryBroadcast(
                c_s_surf_distribution, [f"{domain} {phase_name}particle"]
            )
        else:
            c_s_distribution = c_s

            # x-average the *tertiary* domain.
            # NOTE: not yet implemented. Make 0.5 everywhere
            c_s_xav_distribution = pybamm.FullBroadcast(
                0.5,
                [f"{domain} {phase_name}particle"],
                {
                    "secondary": f"{domain} {phase_name}particle size",
                    "tertiary": "current collector",
                },
            )

            # Surface concentration distribution variables
            c_s_surf_distribution = pybamm.surf(c_s)
            c_s_surf_xav_distribution = pybamm.x_average(c_s_surf_distribution)

        c_s_rav_distribution = pybamm.r_average(c_s_distribution)
        c_s_av_distribution = pybamm.x_average(c_s_rav_distribution)

        variables = {
            f"Average {domain} {phase_name}particle concentration "
            "distribution": c_s_av_distribution / c_scale,
            f"Average {domain} {phase_name}particle concentration "
            "distribution [mol.m-3]": c_s_av_distribution,
            f"{Domain} {phase_name}particle concentration "
            "distribution": c_s_distribution / c_scale,
            f"{Domain} {phase_name}particle concentration distribution "
            "[mol.m-3]": c_s_distribution,
            f"R-averaged {domain} {phase_name}particle concentration "
            "distribution": c_s_rav_distribution / c_scale,
            f"R-averaged {domain} {phase_name}particle concentration distribution "
            "[mol.m-3]": c_s_rav_distribution,
            f"X-averaged {domain} {phase_name}particle concentration "
            "distribution": c_s_xav_distribution / c_scale,
            f"X-averaged {domain} {phase_name}particle concentration distribution "
            "[mol.m-3]": c_s_xav_distribution,
            f"X-averaged {domain} {phase_name}particle surface concentration"
            " distribution": c_s_surf_xav_distribution / c_scale,
            f"X-averaged {domain} {phase_name}particle surface concentration "
            "distribution [mol.m-3]": c_s_surf_xav_distribution,
            f"{Domain} {phase_name}particle surface concentration"
            " distribution": c_s_surf_distribution / c_scale,
            f"{Domain} {phase_name}particle surface stoichiometry"
            " distribution": c_s_surf_distribution / c_scale,
            f"{Domain} {phase_name}particle surface concentration"
            " distribution [mol.m-3]": c_s_surf_distribution,
        }
        return variables

    def _get_standard_flux_distribution_variables(self, N_s):
        """
        Forms standard flux variables that depend on particle size R given
        the flux variable N_s from the distribution submodel.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if [f"{domain} electrode"] in N_s.domains.values():
            # N_s depends on x

            N_s_distribution = N_s
            # x-av the *tertiary* domain
            # NOTE: not yet implemented. Fill with zeros instead
            N_s_xav_distribution = pybamm.FullBroadcast(
                0,
                [f"{domain} {phase_name}particle"],
                {
                    "secondary": f"{domain} {phase_name}particle size",
                    "tertiary": "current collector",
                },
            )
        else:
            N_s_xav_distribution = N_s
            N_s_distribution = pybamm.TertiaryBroadcast(N_s, [f"{domain} electrode"])

        variables = {
            f"X-averaged {domain} {phase_name}particle flux "
            "distribution [mol.m-2.s-1]": N_s_xav_distribution,
            f"{Domain} {phase_name}particle flux "
            "distribution [mol.m-2.s-1]": N_s_distribution,
        }

        return variables

    def _get_standard_diffusivity_variables(self, D_eff):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        variables = {
            f"{Domain} {phase_name}particle effective diffusivity [m2.s-1]": D_eff,
            f"X-averaged {domain} {phase_name}particle effective "
            "diffusivity [m2.s-1]": pybamm.x_average(D_eff),
        }
        return variables

    def _get_standard_diffusivity_distribution_variables(self, D_eff):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        variables = {
            f"{Domain} {phase_name}particle effective diffusivity "
            "distribution [m2.s-1]": D_eff,
            f"X-averaged {domain} {phase_name}particle effective diffusivity "
            "distribution[m2.s-1]": pybamm.x_average(D_eff),
        }

        return variables
