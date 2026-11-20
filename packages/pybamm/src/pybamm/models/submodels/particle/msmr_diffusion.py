#
# Class for particles using the MSMR model
#
import pybamm

from .base_particle import BaseParticle


class MSMRDiffusion(BaseParticle):
    """
    Class for molar conservation in particles within the Multi-Species Multi-Reaction
    framework :footcite:t:`Baker2018`. The thermodynamic model is presented in
    :footcite:t:`Verbrugge2017`, along with parameter values for a number of
    substitutional materials.

    In this submodel, the stoichiometry depends on the potential in the particle and
    the temperature, so dUdT is not used. See `:meth:`pybamm.LithiumIonParameters.dUdT`
    for more explanation.

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
    x_average : bool
        Whether the particle concentration is averaged over the x-direction
    """

    def __init__(self, param, domain, options, phase="primary", x_average=False):
        super().__init__(param, domain, options, phase, x_average)

        pybamm.citations.register("Baker2018")
        pybamm.citations.register("Verbrugge2017")

    def build(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        particle = f"{domain} {phase_name}particle"
        c_max = self.phase_param.c_max

        T = self._broadcast_to_particle(self._coupled_temperature())
        j = self._coupled_interfacial_current_density()
        current = self.coupled_variable("Total current density [A.m-2]")
        # The stoichiometry is computed from the potential by
        # MSMRStoichiometryVariables
        distribution = " distribution" if self.size_distribution else ""
        x = self.coupled_variable(
            f"{self._x_prefix} {phase_name}particle stoichiometry{distribution}",
            domains=self._particle_domains,
        )
        dxdU = self.coupled_variable(
            f"{self._x_prefix} {phase_name}particle differential "
            f"stoichiometry{distribution} [V-1]",
            domains=self._particle_domains,
        )

        # In the MSMR model we solve for the potential in the particles (and
        # particle-size distribution, if applicable), which sets the stoichiometry
        U = pybamm.Variable(
            f"{self._x_prefix} {phase_name}particle potential{distribution} [V]",
            domains=self._particle_domains,
        )
        if self.size_distribution:
            R, f_v_dist, variables = self._get_size_distribution()
            variables.update(self._get_standard_potential_distribution_variables(U))
            # Size-average with the volume-weighted distribution, since the
            # potential sets volume-based quantities (e.g. the total lithium)
            U_electrode = pybamm.Integral(f_v_dist * U, R)
            R_nondim = variables[f"{Domain} {phase_name}particle sizes"]
            R_broad_nondim = pybamm.PrimaryBroadcast(R_nondim, [particle])
        else:
            U.print_name = f"U_{domain[0]}" + ("_xav" if self.x_average else "")
            U_electrode = U
            variables = {}
            if self.x_average:
                R_nondim = 1
            else:
                R_nondim = self.coupled_variable(
                    f"{Domain} {phase_name}particle radius", domains=self._domains()
                )
            R_broad_nondim = R_nondim
        # The standard variables need the potential at every point in the electrode
        if self.x_average:
            U_electrode = pybamm.SecondaryBroadcast(
                U_electrode, [f"{domain} electrode"]
            )
        variables.update(self._get_standard_potential_variables(U_electrode))

        # Note: diffusivity is given as a function of concentration here,
        # not stoichiometry
        D_eff = self._get_effective_diffusivity(x * c_max, T, current)
        f = self.param.F / (self.param.R * T)
        N_s = c_max * x * (1 - x) * f * D_eff * pybamm.grad(U)
        rhs = -(1 / (R_broad_nondim**2)) * pybamm.div(N_s) / c_max / dxdU
        rbc = j * R_nondim / self.param.F / pybamm.surf(c_max * x * (1 - x) * f * D_eff)
        variables.update(
            {
                f"{Domain} {phase_name}particle rhs [V.s-1]": rhs,
                f"{Domain} {phase_name}particle bc [V.m-1]": rbc,
            }
        )
        variables.update(self._get_flux_and_diffusivity_variables(N_s, D_eff))
        self.variables.update(variables)

        self.rhs = {U: rhs}
        self.boundary_conditions = {
            U: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }
        self.initial_conditions = {U: self.phase_param.U_init}

    def _get_standard_potential_variables(self, U):
        """
        A private function to obtain the standard variables which can be derived from
        the potential.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        U_surf = pybamm.surf(U)
        U_surf_av = pybamm.x_average(U_surf)
        U_xav = pybamm.x_average(U)
        U_rav = pybamm.r_average(U)
        U_av = pybamm.r_average(U_xav)
        variables = {
            f"{Domain} {phase_name}particle potential [V]": U,
            f"X-averaged {domain} {phase_name}particle potential [V]": U_xav,
            f"R-averaged {domain} {phase_name}particle potential [V]": U_rav,
            f"Average {domain} {phase_name}particle potential [V]": U_av,
            f"{Domain} {phase_name}particle surface potential [V]": U_surf,
            f"X-averaged {domain} {phase_name}particle "
            "surface potential [V]": U_surf_av,
            f"Minimum {domain} {phase_name}particle potential [V]": pybamm.min(U),
            f"Maximum {domain} {phase_name}particle potential [V]": pybamm.max(U),
            f"Minimum {domain} {phase_name}particle surface potential [V]": pybamm.min(
                U_surf
            ),
            f"Maximum {domain} {phase_name}particle surface potential [V]": pybamm.max(
                U_surf
            ),
        }
        return variables

    def _get_standard_potential_distribution_variables(self, U):
        """
        A private function to obtain the standard variables which can be derived from
        the potential distribution in particle size.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        # Broadcast and x-average when necessary
        if U.domain == [f"{domain} {phase_name}particle size"] and U.domains[
            "secondary"
        ] != [f"{domain} electrode"]:
            # X-avg potential distribution
            U_xav_distribution = pybamm.PrimaryBroadcast(
                U, [f"{domain} {phase_name}particle"]
            )

            # Surface potential distribution variables
            U_surf_xav_distribution = U
            U_surf_distribution = pybamm.SecondaryBroadcast(
                U_surf_xav_distribution, [f"{domain} electrode"]
            )

            # potential distribution in all domains.
            U_distribution = pybamm.PrimaryBroadcast(
                U_surf_distribution, [f"{domain} {phase_name}particle"]
            )
        elif U.domain == [f"{domain} {phase_name}particle"] and (
            U.domains["tertiary"] != [f"{domain} electrode"]
        ):
            # X-avg potential distribution
            U_xav_distribution = U

            # Surface potential distribution variables
            U_surf_xav_distribution = pybamm.surf(U_xav_distribution)
            U_surf_distribution = pybamm.SecondaryBroadcast(
                U_surf_xav_distribution, [f"{domain} electrode"]
            )

            # potential distribution in all domains
            U_distribution = pybamm.TertiaryBroadcast(
                U_xav_distribution, [f"{domain} electrode"]
            )
        elif U.domain == [f"{domain} {phase_name}particle size"] and U.domains[
            "secondary"
        ] == [f"{domain} electrode"]:
            # Surface potential distribution variables
            U_surf_distribution = U
            U_surf_xav_distribution = pybamm.x_average(U)

            # X-avg potential distribution
            U_xav_distribution = pybamm.PrimaryBroadcast(
                U_surf_xav_distribution, [f"{domain} {phase_name}particle"]
            )

            # potential distribution in all domains
            U_distribution = pybamm.PrimaryBroadcast(
                U_surf_distribution, [f"{domain} {phase_name}particle"]
            )
        else:
            U_distribution = U

            # x-average the *tertiary* domain.
            # NOTE: not yet implemented. Make 0.5 everywhere
            U_xav_distribution = pybamm.FullBroadcast(
                0.5,
                [f"{domain} {phase_name}particle"],
                {
                    "secondary": f"{domain} {phase_name}particle size",
                    "tertiary": "current collector",
                },
            )

            # Surface potential distribution variables
            U_surf_distribution = pybamm.surf(U)
            U_surf_xav_distribution = pybamm.x_average(U_surf_distribution)

        U_rav_distribution = pybamm.r_average(U_distribution)
        U_av_distribution = pybamm.x_average(U_rav_distribution)

        variables = {
            f"{Domain} {phase_name}particle potential distribution [V]": U_distribution,
            f"X-averaged {domain} {phase_name}particle potential "
            "distribution [V]": U_xav_distribution,
            f"R-averaged {domain} {phase_name}particle potential "
            "distribution [V]": U_rav_distribution,
            f"Average {domain} {phase_name}particle potential "
            "distribution [V]": U_av_distribution,
            f"{Domain} {phase_name}particle surface potential"
            " distribution [V]": U_surf_distribution,
            f"X-averaged {domain} {phase_name}particle surface potential "
            "distribution [V]": U_surf_xav_distribution,
        }
        return variables


class MSMRStoichiometryVariables(BaseParticle):
    """
    Class for the stoichiometry and concentration variables of a particle in the
    Multi-Species Multi-Reaction framework, computed from the particle potential
    set by :class:`MSMRDiffusion`.

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
    x_average : bool
        Whether the particle potential is averaged over the x-direction. The
        variables are computed from the potential broadcast over the electrode, so
        they do not depend on this.
    """

    def build(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        particle = f"{domain} {phase_name}particle"
        electrode = f"{domain} electrode"

        U = self.coupled_variable(
            f"{Domain} {phase_name}particle potential [V]",
            domain=particle,
            auxiliary_domains={"secondary": electrode, "tertiary": "current collector"},
        )
        T = self.coupled_variable(
            f"{Domain} electrode temperature [K]",
            domain=electrode,
            auxiliary_domains={"secondary": "current collector"},
        )

        # Standard fractional occupancy variables (these are indexed by reaction number)
        variables = self._get_standard_fractional_occupancy_variables(U, T)
        variables.update(
            self._get_standard_differential_fractional_occupancy_variables(U, T)
        )

        # Calculate the (total) stoichiometry from the potential
        x = self.phase_param.x(U, T)
        dxdU = self.phase_param.dxdU(U, T)

        # Standard (total) stoichiometry and concentration variables (size-independent)
        c_max = self.phase_param.c_max
        variables.update(self._get_standard_concentration_variables(x * c_max))
        variables.update(self._get_standard_differential_stoichiometry_variables(dxdU))

        if self.size_distribution:
            U_distribution = self.coupled_variable(
                f"{Domain} {phase_name}particle potential distribution [V]",
                domain=particle,
                auxiliary_domains={
                    "secondary": f"{particle} size",
                    "tertiary": electrode,
                    "quaternary": "current collector",
                },
            )
            T_distribution = pybamm.PrimaryBroadcast(
                pybamm.PrimaryBroadcast(T, f"{particle} size"), particle
            )
            # Calculate the stoichiometry distribution from the potential distribution
            x_distribution = self.phase_param.x(U_distribution, T_distribution)
            dxdU_distribution = self.phase_param.dxdU(U_distribution, T_distribution)

            # Standard stoichiometry and concentration distribution variables
            # (size-dependent)
            variables.update(
                self._get_standard_concentration_distribution_variables(
                    x_distribution * c_max
                )
            )
            variables.update(
                self._get_standard_differential_stoichiometry_distribution_variables(
                    dxdU_distribution
                )
            )
        self.variables.update(variables)

    def _get_standard_fractional_occupancy_variables(self, U, T):
        options = self.options
        domain = self.domain
        d = domain[0]
        variables = {}
        # Loop over all reactions
        N = int(getattr(options, domain)["number of MSMR reactions"])
        for i in range(N):
            x = self.phase_param.x_j(U, T, i)
            x_surf = pybamm.surf(x)
            x_surf_av = pybamm.x_average(x_surf)
            x_xav = pybamm.x_average(x)
            x_rav = pybamm.r_average(x)
            x_av = pybamm.r_average(x_xav)
            variables.update(
                {
                    f"x_{d}_{i}": x,
                    f"X-averaged x_{d}_{i}": x_xav,
                    f"R-averaged x_{d}_{i}": x_rav,
                    f"Average x_{d}_{i}": x_av,
                    f"Surface x_{d}_{i}": x_surf,
                    f"X-averaged surface x_{d}_{i}": x_surf_av,
                }
            )
        return variables

    def _get_standard_differential_fractional_occupancy_variables(self, U, T):
        options = self.options
        domain = self.domain
        d = domain[0]
        variables = {}
        # Loop over all reactions
        N = int(getattr(options, domain)["number of MSMR reactions"])
        for i in range(N):
            dxdU = self.phase_param.dxdU_j(U, T, i)
            dxdU_surf = pybamm.surf(dxdU)
            dxdU_surf_av = pybamm.x_average(dxdU_surf)
            dxdU_xav = pybamm.x_average(dxdU)
            dxdU_rav = pybamm.r_average(dxdU)
            dxdU_av = pybamm.r_average(dxdU_xav)
            variables.update(
                {
                    f"dxdU_{d}_{i}": dxdU,
                    f"X-averaged dxdU_{d}_{i}": dxdU_xav,
                    f"R-averaged dxdU_{d}_{i}": dxdU_rav,
                    f"Average dxdU_{d}_{i}": dxdU_av,
                    f"Surface dxdU_{d}_{i}": dxdU_surf,
                    f"X-averaged surface dxdU_{d}_{i}": dxdU_surf_av,
                }
            )
        return variables

    def _get_standard_differential_stoichiometry_variables(self, dxdU):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        dxdU_surf = pybamm.surf(dxdU)
        dxdU_surf_av = pybamm.x_average(dxdU_surf)
        dxdU_xav = pybamm.x_average(dxdU)
        dxdU_rav = pybamm.r_average(dxdU)
        dxdU_av = pybamm.r_average(dxdU_xav)

        variables = {
            f"{Domain} {phase_name}particle differential stoichiometry [V-1]": dxdU,
            f"X-averaged {domain} {phase_name}particle "
            "differential stoichiometry [V-1]": dxdU_xav,
            f"R-averaged {domain} {phase_name}particle "
            "differential stoichiometry [V-1]": dxdU_rav,
            f"Average {domain} {phase_name}particle differential "
            "stoichiometry [V-1]": dxdU_av,
            f"{Domain} {phase_name}particle surface differential "
            "stoichiometry [V-1]": dxdU_surf,
            f"X-averaged {domain} {phase_name}particle "
            "surface differential stoichiometry [V-1]": dxdU_surf_av,
        }

        return variables

    def _get_standard_differential_stoichiometry_distribution_variables(self, dxdU):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        # Broadcast and x-average when necessary
        if dxdU.domain == [f"{domain} {phase_name}particle size"] and dxdU.domains[
            "secondary"
        ] != [f"{domain} electrode"]:
            # X-avg differential stoichiometry distribution
            dxdU_xav_distribution = pybamm.PrimaryBroadcast(
                dxdU, [f"{domain} {phase_name}particle"]
            )

            # Surface differential stoichiometry distribution variables
            dxdU_surf_xav_distribution = dxdU
            dxdU_surf_distribution = pybamm.SecondaryBroadcast(
                dxdU_surf_xav_distribution, [f"{domain} electrode"]
            )

            # Differential stoichiometry distribution in all domains.
            dxdU_distribution = pybamm.PrimaryBroadcast(
                dxdU_surf_distribution, [f"{domain} {phase_name}particle"]
            )
        elif dxdU.domain == [f"{domain} {phase_name}particle"] and (
            dxdU.domains["tertiary"] != [f"{domain} electrode"]
        ):
            # X-avg differential stoichiometry distribution
            dxdU_xav_distribution = dxdU

            # Surface differential stoichiometry distribution variables
            dxdU_surf_xav_distribution = pybamm.surf(dxdU_xav_distribution)
            dxdU_surf_distribution = pybamm.SecondaryBroadcast(
                dxdU_surf_xav_distribution, [f"{domain} electrode"]
            )

            # Differential stoichiometry distribution in all domains.
            dxdU_distribution = pybamm.TertiaryBroadcast(
                dxdU_xav_distribution, [f"{domain} electrode"]
            )
        elif dxdU.domain == [f"{domain} {phase_name}particle size"] and dxdU.domains[
            "secondary"
        ] == [f"{domain} electrode"]:
            # Surface differential stoichiometry distribution variables
            dxdU_surf_distribution = dxdU
            dxdU_surf_xav_distribution = pybamm.x_average(dxdU)

            # X-avg differential stoichiometry distribution
            dxdU_xav_distribution = pybamm.PrimaryBroadcast(
                dxdU_surf_xav_distribution, [f"{domain} {phase_name}particle"]
            )

            # Differential stoichiometry distribution in all domains
            dxdU_distribution = pybamm.PrimaryBroadcast(
                dxdU_surf_distribution, [f"{domain} {phase_name}particle"]
            )
        else:
            dxdU_distribution = dxdU

            # x-average the *tertiary* domain.
            # NOTE: not yet implemented. Make 0.5 everywhere
            dxdU_xav_distribution = pybamm.FullBroadcast(
                0.5,
                [f"{domain} {phase_name}particle"],
                {
                    "secondary": f"{domain} {phase_name}particle size",
                    "tertiary": "current collector",
                },
            )

            # Surface differential stoichiometry distribution variables
            dxdU_surf_distribution = pybamm.surf(dxdU)
            dxdU_surf_xav_distribution = pybamm.x_average(dxdU_surf_distribution)

        dxdU_rav_distribution = pybamm.r_average(dxdU_distribution)
        dxdU_av_distribution = pybamm.x_average(dxdU_rav_distribution)

        variables = {
            f"{Domain} {phase_name}particle differential stoichiometry distribution "
            "[V-1]": dxdU_distribution,
            f"X-averaged {domain} {phase_name}particle differential stoichiometry "
            "distribution [V-1]": dxdU_xav_distribution,
            f"R-averaged {domain} {phase_name}particle differential stoichiometry "
            "distribution [V-1]": dxdU_rav_distribution,
            f"Average {domain} {phase_name}particle differential stoichiometry "
            "distribution [V-1]": dxdU_av_distribution,
            f"{Domain} {phase_name}particle surface differential stoichiometry"
            " distribution [V-1]": dxdU_surf_distribution,
            f"X-averaged {domain} {phase_name}particle surface differential "
            "stoichiometry distribution [V-1]": dxdU_surf_xav_distribution,
        }
        return variables
