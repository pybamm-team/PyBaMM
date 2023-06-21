#
# Class for single particle with polynomial concentration profile
#
import pybamm

from .polynomial_profile import PolynomialProfile


class XAveragedPolynomialProfile(PolynomialProfile):
    """
    Class for molar conservation in a single x-averaged particle employing Fick's law,
    with an assumed polynomial concentration profile in r. Model equations from
    :footcite:t:`Subramanian2005`.

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

    """

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options, phase)

    def get_fundamental_variables(self):
        domain = self.domain

        variables = {}
        # For all orders we solve an equation for the average concentration
        if self.size_distribution is False:
            c_s_av = pybamm.Variable(
                f"Average {domain} particle concentration [mol.m-3]",
                domain="current collector",
                bounds=(0, self.phase_param.c_max),
                scale=self.phase_param.c_max,
            )
        else:
            c_s_av_distribution = pybamm.Variable(
                f"Average {domain} particle concentration distribution [mol.m-3]",
                domain=f"{domain} particle size",
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, self.phase_param.c_max),
                scale=self.phase_param.c_max,
            )
            # Since concentration does not depend on "x", need a particle-size
            # spatial variable R with only "current collector" as secondary
            # domain
            R = pybamm.SpatialVariable(
                f"R_{domain[0]}",
                domain=[f"{domain} particle size"],
                auxiliary_domains={"secondary": "current collector"},
                coord_sys="cartesian",
            )
            variables = self._get_distribution_variables(R)

            # Standard distribution variables (size-dependent)
            variables.update(
                self._get_standard_concentration_distribution_variables(
                    c_s_av_distribution
                )
            )

            # Standard size-averaged variables. Average concentrations using
            # the volume-weighted distribution since they are volume-based
            # quantities. Necessary for output variables "Total lithium in
            # negative electrode [mol]", etc, to be calculated correctly
            f_v_dist = variables[
                f"X-averaged {domain} volume-weighted particle-size distribution [m-1]"
            ]
            c_s_av = pybamm.Integral(f_v_dist * c_s_av_distribution, R)

        variables.update({f"Average {domain} particle concentration [mol.m-3]": c_s_av})

        # For the fourth order polynomial approximation we also solve an
        # equation for the average concentration gradient. Note: in the original
        # paper this quantity is referred to as the flux, but here we make the
        # distinction between the flux defined as N = -D*dc/dr and the concentration
        # gradient q = dc/dr
        if self.name == "quartic profile":
            q_s_av = pybamm.Variable(
                f"Average {domain} particle concentration gradient [mol.m-4]",
                domain="current collector",
                scale=self.phase_param.c_max / self.phase_param.R_typ,
            )
            variables.update(
                {f"Average {domain} particle concentration gradient [mol.m-4]": q_s_av}
            )

        return variables

    def get_coupled_variables(self, variables):
        domain = self.domain
        param = self.param

        c_s_av = variables[f"Average {domain} particle concentration [mol.m-3]"]
        T_av = variables[f"X-averaged {domain} electrode temperature [K]"]
        R = variables[f"X-averaged {domain} particle radius [m]"]

        if self.name != "uniform profile":
            D_eff_av = self._get_effective_diffusivity(c_s_av, T_av)
            i_boundary_cc = variables["Current collector current density [A.m-2]"]
            a_av = variables[
                f"X-averaged {domain} electrode surface area to volume ratio [m-1]"
            ]
            sgn = 1 if self.domain == "negative" else -1

            j_xav = sgn * i_boundary_cc / (a_av * self.domain_param.L)

        # Set surface concentration based on polynomial order
        if self.name == "uniform profile":
            # The concentration is uniform so the surface value is equal to
            # the average
            c_s_surf_xav = c_s_av
        elif self.name == "quadratic profile":
            # The surface concentration is computed from the average concentration
            # and boundary flux
            # Note 1: here we use the total average interfacial current for the single
            # particle. We explicitly write this as the current density divided by the
            # electrode thickness instead of getting the average current from the
            # interface submodel since the interface submodel requires the surface
            # concentration to be defined first to compute the exchange current density.
            # Explicitly writing out the average interfacial current here avoids
            # KeyErrors due to variables not being set in the "right" order.
            # Note 2: the concentration, c, inside the diffusion coefficient, D, here
            # should really be the surface value, but this requires solving a nonlinear
            # equation for c_surf (if the diffusion coefficient is nonlinear), adding
            # an extra algebraic equation to solve. For now, using the average c is an
            # ok approximation and means the SPM(e) still gives a system of ODEs rather
            # than DAEs.
            c_s_surf_xav = c_s_av - (j_xav * R / 5 / param.F / D_eff_av)
        elif self.name == "quartic profile":
            # The surface concentration is computed from the average concentration,
            # the average concentration gradient and the boundary flux (see notes
            # for the quadratic profile)
            # eq 31 of Subramanian2005
            q_s_av = variables[
                f"Average {domain} particle concentration gradient [mol.m-4]"
            ]
            c_s_surf_xav = c_s_av + R / 35 * (8 * q_s_av - (j_xav / param.F / D_eff_av))

        # Set concentration depending on polynomial order
        # Since c_s_xav doesn't depend on x, we need to define a spatial
        # variable r which only has "... particle" and "current
        # collector" as domains
        r = pybamm.SpatialVariable(
            f"r_{domain[0]}",
            domain=[f"{domain} particle"],
            auxiliary_domains={"secondary": "current collector"},
            coord_sys="spherical polar",
        )
        if self.name == "uniform profile":
            # The concentration is uniform
            A = c_s_av
            B = pybamm.PrimaryBroadcast(0, "current collector")
            C = pybamm.PrimaryBroadcast(0, "current collector")
        elif self.name == "quadratic profile":
            # The concentration is given by c = A + B*r**2
            # eqs 11-12 in Subramanian2005
            A = 5 / 2 * c_s_av - 3 / 2 * c_s_surf_xav
            B = (5 / 2) * (c_s_surf_xav - c_s_av)
            C = pybamm.PrimaryBroadcast(0, "current collector")
        elif self.name == "quartic profile":
            # The concentration is given by c = A + B*r**2 + C*r**4
            # eqs 24-26 in Subramanian2005
            A = 39 / 4 * c_s_surf_xav - 3 * q_s_av * R - 35 / 4 * c_s_av
            B = -35 * c_s_surf_xav + 10 * q_s_av * R + 35 * c_s_av
            C = 105 / 4 * c_s_surf_xav - 7 * q_s_av * R - 105 / 4 * c_s_av

        A = pybamm.PrimaryBroadcast(A, [f"{domain} particle"])
        B = pybamm.PrimaryBroadcast(B, [f"{domain} particle"])
        C = pybamm.PrimaryBroadcast(C, [f"{domain} particle"])
        c_s_xav = A + B * r**2 / R**2 + C * r**4 / R**4
        c_s = pybamm.SecondaryBroadcast(c_s_xav, [f"{domain} electrode"])
        c_s_surf = pybamm.PrimaryBroadcast(c_s_surf_xav, [f"{domain} electrode"])

        # Set flux based on polynomial order
        if self.name != "uniform profile":
            T_xav = pybamm.PrimaryBroadcast(T_av, [f"{domain} particle"])
            D_eff_xav = self._get_effective_diffusivity(c_s_xav, T_xav)
            D_eff = pybamm.SecondaryBroadcast(D_eff_xav, [f"{domain} electrode"])
            variables.update(self._get_standard_diffusivity_variables(D_eff))
        if self.name == "uniform profile":
            # The flux is zero since there is no concentration gradient
            N_s_xav = pybamm.FullBroadcastToEdges(
                0, f"{domain} particle", "current collector"
            )
        elif self.name == "quadratic profile":
            # The flux may be computed directly from the polynomial for c
            N_s_xav = -D_eff_xav * 5 * (c_s_surf_xav - c_s_av) * r / R**2
        elif self.name == "quartic profile":
            q_s_av = variables[
                f"Average {domain} particle concentration gradient [mol.m-4]"
            ]
            # The flux may be computed directly from the polynomial for c
            N_s_xav = -D_eff_xav * (
                (-70 * c_s_surf_xav + 20 * q_s_av * R + 70 * c_s_av) * r / R**2
                + (105 * c_s_surf_xav - 28 * q_s_av * R - 105 * c_s_av)
                * (r**3 / R**4)
            )

        N_s = pybamm.SecondaryBroadcast(N_s_xav, [f"{domain} electrode"])

        variables.update(
            self._get_standard_concentration_variables(
                c_s, c_s_av=c_s_av, c_s_surf=c_s_surf
            )
        )
        variables.update(self._get_standard_flux_variables(N_s))

        return variables

    def set_rhs(self, variables):
        # Note: we have to use `pybamm.source(rhs, var)` in the rhs dict so that
        # the scalar source term gets multplied by the correct mass matrix when
        # using this model with 2D current collectors with the finite element
        # method (see #1399)
        domain = self.domain
        param = self.param

        if self.size_distribution is False:
            c_s_av = variables[f"Average {domain} particle concentration [mol.m-3]"]
            j_xav = variables[
                f"X-averaged {domain} electrode interfacial current density [A.m-2]"
            ]
            R = variables[f"X-averaged {domain} particle radius [m]"]
        else:
            c_s_av = variables[
                f"Average {domain} particle concentration distribution [mol.m-3]"
            ]
            j_xav = variables[
                f"X-averaged {domain} electrode interfacial "
                "current density distribution [A.m-2]"
            ]
            R = variables[f"X-averaged {domain} particle sizes [m]"]

        # eq 15 of Subramanian2005
        # equivalent to dcdt = -i_cc / (eps * F * L)
        dcdt = -3 * j_xav / param.F / R

        if self.size_distribution is False:
            self.rhs = {c_s_av: pybamm.source(dcdt, c_s_av)}
        else:
            self.rhs = {c_s_av: dcdt}

        if self.name == "quartic profile":
            # We solve an extra ODE for the average particle concentration gradient
            q_s_av = variables[
                f"Average {domain} particle concentration gradient [mol.m-4]"
            ]
            D_eff_xav = variables[
                f"X-averaged {domain} particle effective diffusivity [m2.s-1]"
            ]

            # eq 30 of Subramanian2005
            dqdt = (
                -30 * pybamm.surf(D_eff_xav) * q_s_av / R**2
                - 45 / 2 * j_xav / param.F / R**2
            )
            self.rhs[q_s_av] = pybamm.source(dqdt, q_s_av)

    def set_algebraic(self, variables):
        pass

    def set_initial_conditions(self, variables):
        """
        For single or x-averaged particle models, initial conditions can't depend on x
        or r so we take the r- and x-average of the initial conditions.
        """
        domain = self.domain
        c_init = pybamm.x_average(pybamm.r_average(self.phase_param.c_init))

        if self.size_distribution is False:
            c_s_av = variables[f"Average {domain} particle concentration [mol.m-3]"]
        else:
            c_s_av = variables[
                f"Average {domain} particle concentration distribution [mol.m-3]"
            ]
            c_init = c_init = pybamm.PrimaryBroadcast(c_init, f"{domain} particle size")

        self.initial_conditions = {c_s_av: c_init}
        if self.name == "quartic profile":
            # We also need to provide an initial condition for the average
            # concentration gradient
            q_s_av = variables[
                f"Average {domain} particle concentration gradient [mol.m-4]"
            ]
            self.initial_conditions.update({q_s_av: 0})
