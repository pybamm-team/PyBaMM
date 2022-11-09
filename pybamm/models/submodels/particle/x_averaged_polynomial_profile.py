#
# Class for single particle with polynomial concentration profile
#
import pybamm

from .polynomial_profile import PolynomialProfile


class XAveragedPolynomialProfile(PolynomialProfile):
    """
    Class for molar conservation in a single x-averaged particle employing Fick's law,
    with an assumed polynomial concentration profile in r. Model equations from [1]_.

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

    References
    ----------
    .. [1] VR Subramanian, VD Diwakar and D Tapriyal. “Efficient Macro-Micro Scale
           Coupled Modeling of Batteries”. Journal of The Electrochemical Society,
           152(10):A2002-A2008, 2005

    **Extends:** :class:`pybamm.particle.PolynomialProfile`
    """

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options, phase)

    def get_fundamental_variables(self):
        domain = self.domain

        variables = {}
        # For all orders we solve an equation for the average concentration
        if self.size_distribution is False:
            c_s_av = pybamm.Variable(
                f"Average {domain} particle concentration",
                domain="current collector",
                bounds=(0, 1),
            )
        else:
            c_s_av_distribution = pybamm.Variable(
                f"Average {domain} particle concentration distribution",
                domain=f"{domain} particle size",
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, 1),
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
                f"X-averaged {domain} volume-weighted particle-size distribution"
            ]
            c_s_av = pybamm.Integral(f_v_dist * c_s_av_distribution, R)

        variables.update({f"Average {domain} particle concentration": c_s_av})

        # For the fourth order polynomial approximation we also solve an
        # equation for the average concentration gradient. Note: in the original
        # paper this quantity is referred to as the flux, but here we make the
        # distinction between the flux defined as N = -D*dc/dr and the concentration
        # gradient q = dc/dr
        if self.name == "quartic profile":
            q_s_av = pybamm.Variable(
                f"Average {domain} particle concentration gradient",
                domain="current collector",
            )
            variables.update(
                {f"Average {domain} particle concentration gradient": q_s_av}
            )

        return variables

    def get_coupled_variables(self, variables):
        domain = self.domain
        phase_param = self.phase_param
        phase_param = self.phase_param

        c_s_av = variables[f"Average {domain} particle concentration"]
        T_av = variables[f"X-averaged {domain} electrode temperature"]

        if self.name != "uniform profile":
            D_eff_av = self._get_effective_diffusivity(c_s_av, T_av)
            i_boundary_cc = variables["Current collector current density"]
            a_av = variables[
                f"X-averaged {domain} electrode surface area to volume ratio"
            ]
            sgn = 1 if self.domain == "negative" else -1

            j_xav = sgn * i_boundary_cc / (a_av * self.domain_param.l)

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
            c_s_surf_xav = c_s_av - phase_param.C_diff * (
                j_xav / 5 / phase_param.a_R / phase_param.gamma / D_eff_av
            )
        elif self.name == "quartic profile":
            # The surface concentration is computed from the average concentration,
            # the average concentration gradient and the boundary flux (see notes
            # for the quadratic profile)
            q_s_av = variables[f"Average {domain} particle concentration gradient"]
            c_s_surf_xav = (
                c_s_av
                + 8 * q_s_av / 35
                - phase_param.C_diff
                * (j_xav / 35 / phase_param.a_R / phase_param.gamma / D_eff_av)
            )

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
            A = (1 / 2) * (5 * c_s_av - 3 * c_s_surf_xav)
            B = (5 / 2) * (c_s_surf_xav - c_s_av)
            C = pybamm.PrimaryBroadcast(0, "current collector")
        elif self.name == "quartic profile":
            # The concentration is given by c = A + B*r**2 + C*r**4
            A = 39 * c_s_surf_xav / 4 - 3 * q_s_av - 35 * c_s_av / 4
            B = -35 * c_s_surf_xav + 10 * q_s_av + 35 * c_s_av
            C = 105 * c_s_surf_xav / 4 - 7 * q_s_av - 105 * c_s_av / 4

        A = pybamm.PrimaryBroadcast(A, [f"{domain} particle"])
        B = pybamm.PrimaryBroadcast(B, [f"{domain} particle"])
        C = pybamm.PrimaryBroadcast(C, [f"{domain} particle"])
        c_s_xav = A + B * r**2 + C * r**4
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
            N_s_xav = -D_eff_xav * 5 * (c_s_surf_xav - c_s_av) * r
        elif self.name == "quartic profile":
            q_s_av = variables[f"Average {domain} particle concentration gradient"]
            # The flux may be computed directly from the polynomial for c
            N_s_xav = -D_eff_xav * (
                (-70 * c_s_surf_xav + 20 * q_s_av + 70 * c_s_av) * r
                + (105 * c_s_surf_xav - 28 * q_s_av - 105 * c_s_av) * r**3
            )

        N_s = pybamm.SecondaryBroadcast(N_s_xav, [f"{domain} electrode"])

        variables.update(
            self._get_standard_concentration_variables(
                c_s, c_s_av=c_s_av, c_s_surf=c_s_surf
            )
        )
        variables.update(self._get_standard_flux_variables(N_s))
        variables.update(self._get_total_concentration_variables(variables))

        return variables

    def set_rhs(self, variables):
        # Note: we have to use `pybamm.source(rhs, var)` in the rhs dict so that
        # the scalar source term gets multplied by the correct mass matrix when
        # using this model with 2D current collectors with the finite element
        # method (see #1399)
        domain = self.domain
        phase_param = self.phase_param
        phase_param = self.phase_param

        if self.size_distribution is False:
            c_s_av = variables[f"Average {domain} particle concentration"]
            j_xav = variables[
                f"X-averaged {domain} electrode interfacial current density"
            ]
        else:
            c_s_av = variables[f"Average {domain} particle concentration distribution"]
            j_xav = variables[
                f"X-averaged {domain} electrode interfacial "
                "current density distribution"
            ]

        dcdt = -3 * j_xav / phase_param.a_R / phase_param.gamma

        if self.size_distribution is False:
            self.rhs = {c_s_av: pybamm.source(dcdt, c_s_av)}
        else:
            self.rhs = {c_s_av: dcdt}

        if self.name == "quartic profile":
            # We solve an extra ODE for the average particle concentration gradient
            q_s_av = variables[f"Average {domain} particle concentration gradient"]
            D_eff_xav = variables[f"X-averaged {domain} particle effective diffusivity"]

            self.rhs.update(
                {
                    q_s_av: pybamm.source(
                        -30 * pybamm.surf(D_eff_xav) * q_s_av / phase_param.C_diff
                        - 45 * j_xav / phase_param.a_R / phase_param.gamma / 2,
                        q_s_av,
                    )
                }
            )

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
            c_s_av = variables[f"Average {domain} particle concentration"]
        else:
            c_s_av = variables[f"Average {domain} particle concentration distribution"]
            c_init = c_init = pybamm.PrimaryBroadcast(c_init, f"{domain} particle size")

        self.initial_conditions = {c_s_av: c_init}
        if self.name == "quartic profile":
            # We also need to provide an initial condition for the average
            # concentration gradient
            q_s_av = variables[f"Average {domain} particle concentration gradient"]
            self.initial_conditions.update({q_s_av: 0})
