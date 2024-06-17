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
        super().__init__(param, domain, options, phase)
        self.x_average = x_average

        pybamm.citations.register("Baker2018")
        pybamm.citations.register("Verbrugge2017")

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        variables = {}

        # Define "particle" potential variables. In the MSMR model, we solve for the
        # potential as a function of position within the electrode and particles (and
        # particle-size distribution, if applicable). The potential is then used to
        # calculate the stoichiometry, which is used to calculate the particle
        # concentration.
        if self.size_distribution is False:
            if self.x_average is False:
                U = pybamm.Variable(
                    f"{Domain} {phase_name}particle potential [V]",
                    f"{domain} {phase_name}particle",
                    auxiliary_domains={
                        "secondary": f"{domain} electrode",
                        "tertiary": "current collector",
                    },
                )
                U.print_name = f"U_{domain[0]}"
            else:
                U_xav = pybamm.Variable(
                    f"X-averaged {domain} {phase_name}particle potential [V]",
                    f"{domain} {phase_name}particle",
                    auxiliary_domains={"secondary": "current collector"},
                )
                U_xav.print_name = f"U_{domain[0]}_xav"
                U = pybamm.SecondaryBroadcast(U_xav, f"{domain} electrode")
        else:
            if self.x_average is False:
                U_distribution = pybamm.Variable(
                    f"{Domain} {phase_name}particle potential distribution [V]",
                    domain=f"{domain} {phase_name}particle",
                    auxiliary_domains={
                        "secondary": f"{domain} {phase_name}particle size",
                        "tertiary": f"{domain} electrode",
                        "quaternary": "current collector",
                    },
                )
                R = pybamm.SpatialVariable(
                    f"R_{domain[0]}",
                    domain=[f"{domain} {phase_name}particle size"],
                    auxiliary_domains={
                        "secondary": f"{domain} electrode",
                        "tertiary": "current collector",
                    },
                    coord_sys="cartesian",
                )
                variables = self._get_distribution_variables(R)
                f_v_dist = variables[
                    f"{Domain} volume-weighted {phase_name}"
                    "particle-size distribution [m-1]"
                ]
            else:
                U_distribution = pybamm.Variable(
                    f"X-averaged {domain} {phase_name}particle "
                    "potential distribution [V]",
                    domain=f"{domain} {phase_name}particle",
                    auxiliary_domains={
                        "secondary": f"{domain} {phase_name}particle size",
                        "tertiary": "current collector",
                    },
                )
                R = pybamm.SpatialVariable(
                    f"R_{domain[0]}",
                    domain=[f"{domain} {phase_name}particle size"],
                    auxiliary_domains={"secondary": "current collector"},
                    coord_sys="cartesian",
                )
                variables = self._get_distribution_variables(R)
                f_v_dist = variables[
                    f"X-averaged {domain} volume-weighted {phase_name}"
                    "particle-size distribution [m-1]"
                ]

            # Standard potential distribution_variables
            variables.update(
                self._get_standard_potential_distribution_variables(U_distribution)
            )

            # Standard size-averaged variables. Average potentials using
            # the volume-weighted distribution since they are volume-based
            # quantities. Necessary for output variables "Total lithium in
            # negative electrode [mol]", etc, to be calculated correctly
            U = pybamm.Integral(f_v_dist * U_distribution, R)
            if self.x_average is True:
                U = pybamm.SecondaryBroadcast(U, [f"{domain} electrode"])

        # Standard potential variables
        variables.update(self._get_standard_potential_variables(U))

        return variables

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        param = self.param

        if self.size_distribution is False:
            if self.x_average is False:
                x = variables[f"{Domain} {phase_name}particle stoichiometry"]
                dxdU = variables[
                    f"{Domain} {phase_name}particle differential stoichiometry [V-1]"
                ]
                U = variables[f"{Domain} {phase_name}particle potential [V]"]
                T = pybamm.PrimaryBroadcast(
                    variables[f"{Domain} electrode temperature [K]"],
                    [f"{domain} {phase_name}particle"],
                )
                R_nondim = variables[f"{Domain} {phase_name}particle radius"]
                j = variables[
                    f"{Domain} electrode {phase_name}"
                    "interfacial current density [A.m-2]"
                ]
            else:
                x = variables[f"X-averaged {domain} {phase_name}particle stoichiometry"]
                dxdU = variables[
                    f"X-averaged {domain} {phase_name}particle differential "
                    "stoichiometry [V-1]"
                ]
                U = variables[f"X-averaged {domain} {phase_name}particle potential [V]"]
                T = pybamm.PrimaryBroadcast(
                    variables[f"X-averaged {domain} electrode temperature [K]"],
                    [f"{domain} {phase_name}particle"],
                )
                R_nondim = 1
                j = variables[
                    f"X-averaged {domain} electrode {phase_name}"
                    "interfacial current density [A.m-2]"
                ]
            R_broad_nondim = R_nondim
        else:
            R_nondim = variables[f"{Domain} {phase_name}particle sizes"]
            R_broad_nondim = pybamm.PrimaryBroadcast(
                R_nondim, [f"{domain} {phase_name}particle"]
            )
            if self.x_average is False:
                x = variables[
                    f"{Domain} {phase_name}particle stoichiometry distribution"
                ]
                dxdU = variables[
                    f"{Domain} {phase_name}particle differential stoichiometry "
                    "distribution [V-1]"
                ]
                U = variables[
                    f"{Domain} {phase_name}particle potential distribution [V]"
                ]
                # broadcast T to "particle size" domain then again into "particle"
                T = pybamm.PrimaryBroadcast(
                    variables[f"{Domain} electrode temperature [K]"],
                    [f"{domain} {phase_name}particle size"],
                )
                T = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle"])
                j = variables[
                    f"{Domain} electrode {phase_name}interfacial "
                    "current density distribution [A.m-2]"
                ]
            else:
                x = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "stoichiometry distribution"
                ]
                dxdU = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "differential stoichiometry distribution [V-1]"
                ]
                U = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "potential distribution [V]"
                ]
                # broadcast to "particle size" domain then again into "particle"
                T = pybamm.PrimaryBroadcast(
                    variables[f"X-averaged {domain} electrode temperature [K]"],
                    [f"{domain} {phase_name}particle size"],
                )
                T = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle"])
                j = variables[
                    f"X-averaged {domain} electrode {phase_name}interfacial "
                    "current density distribution [A.m-2]"
                ]

        # Note: diffusivity is given as a function of concentration here,
        # not stoichiometry
        c_max = self.phase_param.c_max
        current = variables["Total current density [A.m-2]"]
        D_eff = self._get_effective_diffusivity(x * c_max, T, current)
        f = self.param.F / (self.param.R * T)
        N_s = c_max * x * (1 - x) * f * D_eff * pybamm.grad(U)
        variables.update(
            {
                f"{Domain} {phase_name}particle rhs [V.s-1]": -(1 / (R_broad_nondim**2))
                * pybamm.div(N_s)
                / c_max
                / dxdU,
                f"{Domain} {phase_name}particle bc [V.m-1]": j
                * R_nondim
                / param.F
                / pybamm.surf(c_max * x * (1 - x) * f * D_eff),
            }
        )

        if self.size_distribution is True:
            # Size-dependent flux variables
            variables.update(
                self._get_standard_diffusivity_distribution_variables(D_eff)
            )
            variables.update(self._get_standard_flux_distribution_variables(N_s))
            # Size-averaged flux variables
            R = variables[f"{Domain} {phase_name}particle sizes [m]"]
            f_a_dist = self.phase_param.f_a_dist(R)
            D_eff = pybamm.Integral(f_a_dist * D_eff, R)
            N_s = pybamm.Integral(f_a_dist * N_s, R)

        if self.x_average is True:
            D_eff = pybamm.SecondaryBroadcast(D_eff, [f"{domain} electrode"])
            N_s = pybamm.SecondaryBroadcast(N_s, [f"{domain} electrode"])

        variables.update(self._get_standard_diffusivity_variables(D_eff))
        variables.update(self._get_standard_flux_variables(N_s))

        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.size_distribution is False:
            if self.x_average is False:
                U = variables[f"{Domain} {phase_name}particle potential [V]"]
            else:
                U = variables[f"X-averaged {domain} {phase_name}particle potential [V]"]
        else:
            if self.x_average is False:
                U = variables[
                    f"{Domain} {phase_name}particle potential distribution [V]"
                ]
            else:
                U = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "potential distribution [V]"
                ]
        self.rhs = {U: variables[f"{Domain} {phase_name}particle rhs [V.s-1]"]}

    def set_boundary_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.size_distribution is False:
            if self.x_average is False:
                U = variables[f"{Domain} {phase_name}particle potential [V]"]
            else:
                U = variables[f"X-averaged {domain} {phase_name}particle potential [V]"]
        else:
            if self.x_average is False:
                U = variables[
                    f"{Domain} {phase_name}particle potential distribution [V]"
                ]
            else:
                U = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "potential distribution [V]"
                ]

        rbc = variables[f"{Domain} {phase_name}particle bc [V.m-1]"]
        self.boundary_conditions = {
            U: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        U_init = self.phase_param.U_init
        if self.size_distribution is False:
            if self.x_average is False:
                U = variables[f"{Domain} {phase_name}particle potential [V]"]
            else:
                U = variables[f"X-averaged {domain} {phase_name}particle potential [V]"]
        else:
            if self.x_average is False:
                U = variables[
                    f"{Domain} {phase_name}particle potential distribution [V]"
                ]
            else:
                U = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "potential distribution [V]"
                ]
        self.initial_conditions = {U: U_init}

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
            f"Minimum {domain} {phase_name}particle "
            "surface potential [V]": pybamm.min(U_surf),
            f"Maximum {domain} {phase_name}particle "
            "surface potential [V]": pybamm.max(U_surf),
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
    def __init__(self, param, domain, options, phase="primary", x_average=False):
        super().__init__(param, domain, options, phase)
        self.x_average = x_average

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        U = variables[f"{Domain} {phase_name}particle potential [V]"]
        T = variables[f"{Domain} electrode temperature [K]"]

        # Standard fractional occupancy variables (these are indexed by reaction number)
        variables.update(self._get_standard_fractional_occupancy_variables(U, T))
        variables.update(
            self._get_standard_differential_fractional_occupancy_variables(U, T)
        )

        # Calculate the (total) stoichiometry from the potential
        x = self.phase_param.x(U, T)
        dxdU = self.phase_param.dxdU(U, T)

        # Standard (total) stoichiometry and concentration variables (size-independent)
        c_max = self.phase_param.c_max
        c_s = x * c_max
        variables.update(self._get_standard_concentration_variables(c_s))
        variables.update(self._get_standard_differential_stoichiometry_variables(dxdU))

        if self.size_distribution is True:
            U_distribution = variables[
                f"{Domain} {phase_name}particle potential distribution [V]"
            ]
            T = variables[f"{Domain} electrode temperature [K]"]
            T_distribution = pybamm.PrimaryBroadcast(
                pybamm.PrimaryBroadcast(T, f"{domain} particle size"),
                f"{domain} particle",
            )
            # Calculate the stoichiometry distribution from the potential distribution
            x_distribution = self.phase_param.x(U_distribution, T_distribution)
            dxdU_distribution = self.phase_param.dxdU(U_distribution, T_distribution)

            # Standard stoichiometry and concentration distribution variables
            # (size-dependent)
            c_s_distribution = x_distribution * c_max
            variables.update(
                self._get_standard_concentration_distribution_variables(
                    c_s_distribution
                )
            )
            variables.update(
                self._get_standard_differential_stoichiometry_distribution_variables(
                    dxdU_distribution
                )
            )
        return variables

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
