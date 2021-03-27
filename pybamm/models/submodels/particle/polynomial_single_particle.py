#
# Class for single particle with polynomial concentration profile
#
import pybamm

from .base_particle import BaseParticle


class PolynomialSingleParticle(BaseParticle):
    """
    Class for molar conservation in a single x-averaged particle with
    an assumed polynomial concentration profile in r. Model equations from [1]_.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    name : str
        The name of the polynomial approximation to be used. Can be "uniform
        profile", "quadratic profile" or "quartic profile".

    References
    ----------
    .. [1] VR Subramanian, VD Diwakar and D Tapriyal. “Efficient Macro-Micro Scale
           Coupled Modeling of Batteries”. Journal of The Electrochemical Society,
           152(10):A2002-A2008, 2005

    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param, domain, name):
        super().__init__(param, domain)
        self.name = name

        pybamm.citations.register("Subramanian2005")

    def get_fundamental_variables(self):
        # For all orders we solve an equation for the average concentration
        if self.domain == "Negative":
            c_s_rxav = pybamm.standard_variables.c_s_n_rxav

        elif self.domain == "Positive":
            c_s_rxav = pybamm.standard_variables.c_s_p_rxav

        variables = {
            "Average " + self.domain.lower() + " particle concentration": c_s_rxav
        }

        # For the fourth order polynomial approximation we also solve an
        # equation for the average concentration gradient. Note: in the original
        # paper this quantity is referred to as the flux, but here we make the
        # distinction between the flux defined as N = -D*dc/dr and the concentration
        # gradient q = dc/dr
        if self.name == "quartic profile":
            if self.domain == "Negative":
                q_s_rxav = pybamm.standard_variables.q_s_n_rxav
            elif self.domain == "Positive":
                q_s_rxav = pybamm.standard_variables.q_s_p_rxav
            variables.update(
                {
                    "Average "
                    + self.domain.lower()
                    + " particle concentration gradient": q_s_rxav
                }
            )

        return variables

    def get_coupled_variables(self, variables):
        c_s_rxav = variables[
            "Average " + self.domain.lower() + " particle concentration"
        ]
        i_boundary_cc = variables["Current collector current density"]
        T_xav = pybamm.PrimaryBroadcast(
            variables["X-averaged " + self.domain.lower() + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )

        # Set surface concentration based on polynomial order
        if self.name == "uniform profile":
            # The concentration is uniform so the surface value is equal to
            # the average
            c_s_surf_xav = c_s_rxav
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
            if self.domain == "Negative":
                j_xav = i_boundary_cc / self.param.l_n
                c_s_surf_xav = c_s_rxav - self.param.C_n * (
                    j_xav
                    / 5
                    / self.param.a_R_n
                    / self.param.D_n(c_s_rxav, pybamm.surf(T_xav))
                )

            if self.domain == "Positive":
                j_xav = -i_boundary_cc / self.param.l_p
                c_s_surf_xav = c_s_rxav - self.param.C_p * (
                    j_xav
                    / 5
                    / self.param.a_R_p
                    / self.param.gamma_p
                    / self.param.D_p(c_s_rxav, pybamm.surf(T_xav))
                )
        elif self.name == "quartic profile":
            # The surface concentration is computed from the average concentration,
            # the average concentration gradient and the boundary flux (see notes
            # for the case order=2)
            q_s_rxav = variables[
                "Average " + self.domain.lower() + " particle concentration gradient"
            ]
            if self.domain == "Negative":
                j_xav = i_boundary_cc / self.param.l_n
                c_s_surf_xav = (
                    c_s_rxav
                    + 8 * q_s_rxav / 35
                    - self.param.C_n
                    * (
                        j_xav
                        / 35
                        / self.param.a_R_n
                        / self.param.D_n(c_s_rxav, pybamm.surf(T_xav))
                    )
                )

            if self.domain == "Positive":
                j_xav = -i_boundary_cc / self.param.l_p
                c_s_surf_xav = (
                    c_s_rxav
                    + 8 * q_s_rxav / 35
                    - self.param.C_p
                    * (
                        j_xav
                        / 35
                        / self.param.a_R_p
                        / self.param.gamma_p
                        / self.param.D_p(c_s_rxav, pybamm.surf(T_xav))
                    )
                )

        # Set concentration depending on polynomial order
        if self.name == "uniform profile":
            # The concentration is uniform
            c_s_xav = pybamm.PrimaryBroadcast(
                c_s_rxav, [self.domain.lower() + " particle"]
            )
        elif self.name == "quadratic profile":
            # The concentration is given by c = A + B*r**2
            A = pybamm.PrimaryBroadcast(
                (1 / 2) * (5 * c_s_rxav - 3 * c_s_surf_xav),
                [self.domain.lower() + " particle"],
            )
            B = pybamm.PrimaryBroadcast(
                (5 / 2) * (c_s_surf_xav - c_s_rxav), [self.domain.lower() + " particle"]
            )
            if self.domain == "Negative":
                # Since c_s_xav doesn't depend on x, we need to define a spatial
                # variable r which only has "negative particle" and "current
                # collector" as domains
                r = pybamm.SpatialVariable(
                    "r_n",
                    domain=["negative particle"],
                    auxiliary_domains={"secondary": "current collector"},
                    coord_sys="spherical polar",
                )
                c_s_xav = A + B * r ** 2
            if self.domain == "Positive":
                # Since c_s_xav doesn't depend on x, we need to define a spatial
                # variable r which only has "positive particle" and "current
                # collector" as domains
                r = pybamm.SpatialVariable(
                    "r_p",
                    domain=["positive particle"],
                    auxiliary_domains={"secondary": "current collector"},
                    coord_sys="spherical polar",
                )
                c_s_xav = A + B * r ** 2

        elif self.name == "quartic profile":
            # The concentration is given by c = A + B*r**2 + C*r**4
            A = pybamm.PrimaryBroadcast(
                39 * c_s_surf_xav / 4 - 3 * q_s_rxav - 35 * c_s_rxav / 4,
                [self.domain.lower() + " particle"],
            )
            B = pybamm.PrimaryBroadcast(
                -35 * c_s_surf_xav + 10 * q_s_rxav + 35 * c_s_rxav,
                [self.domain.lower() + " particle"],
            )
            C = pybamm.PrimaryBroadcast(
                105 * c_s_surf_xav / 4 - 7 * q_s_rxav - 105 * c_s_rxav / 4,
                [self.domain.lower() + " particle"],
            )
            if self.domain == "Negative":
                # Since c_s_xav doesn't depend on x, we need to define a spatial
                # variable r which only has "negative particle" and "current
                # collector" as domains
                r = pybamm.SpatialVariable(
                    "r_n",
                    domain=["negative particle"],
                    auxiliary_domains={"secondary": "current collector"},
                    coord_sys="spherical polar",
                )
                c_s_xav = A + B * r ** 2 + C * r ** 4
            if self.domain == "Positive":
                # Since c_s_xav doesn't depend on x, we need to define a spatial
                # variable r which only has "positive particle" and "current
                # collector" as domains
                r = pybamm.SpatialVariable(
                    "r_p",
                    domain=["positive particle"],
                    auxiliary_domains={"secondary": "current collector"},
                    coord_sys="spherical polar",
                )
                c_s_xav = A + B * r ** 2 + C * r ** 4

        c_s = pybamm.SecondaryBroadcast(c_s_xav, [self.domain.lower() + " electrode"])
        c_s_surf = pybamm.PrimaryBroadcast(
            c_s_surf_xav, [self.domain.lower() + " electrode"]
        )

        # Set flux based on polynomial order
        if self.name == "uniform profile":
            # The flux is zero since there is no concentration gradient
            N_s_xav = pybamm.FullBroadcastToEdges(
                0, self.domain.lower() + " particle", "current collector"
            )
        elif self.name == "quadratic profile":
            # The flux may be computed directly from the polynomial for c
            if self.domain == "Negative":
                N_s_xav = (
                    -self.param.D_n(c_s_xav, T_xav) * 5 * (c_s_surf_xav - c_s_rxav) * r
                )
            if self.domain == "Positive":
                N_s_xav = (
                    -self.param.D_p(c_s_xav, T_xav) * 5 * (c_s_surf_xav - c_s_rxav) * r
                )
        elif self.name == "quartic profile":
            q_s_rxav = variables[
                "Average " + self.domain.lower() + " particle concentration gradient"
            ]
            # The flux may be computed directly from the polynomial for c
            if self.domain == "Negative":
                N_s_xav = -self.param.D_n(c_s_xav, T_xav) * (
                    (-70 * c_s_surf_xav + 20 * q_s_rxav + 70 * c_s_rxav) * r
                    + (105 * c_s_surf_xav - 28 * q_s_rxav - 105 * c_s_rxav) * r ** 3
                )
            elif self.domain == "Positive":
                N_s_xav = -self.param.D_p(c_s_xav, T_xav) * (
                    (-70 * c_s_surf_xav + 20 * q_s_rxav + 70 * c_s_rxav) * r
                    + (105 * c_s_surf_xav - 28 * q_s_rxav - 105 * c_s_rxav) * r ** 3
                )

        N_s = pybamm.SecondaryBroadcast(N_s_xav, [self._domain.lower() + " electrode"])

        variables.update(
            self._get_standard_concentration_variables(
                c_s, c_s_av=c_s_rxav, c_s_surf=c_s_surf
            )
        )
        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))
        variables.update(self._get_total_concentration_variables(variables))

        return variables

    def set_rhs(self, variables):
        # Note: we have to use `pybamm.source(rhs, var)` in the rhs dict so that
        # the scalar source term gets multplied by the correct mass matrix when
        # using this model with 2D current collectors with the finite element
        # method (see #1399)

        c_s_rxav = variables[
            "Average " + self.domain.lower() + " particle concentration"
        ]
        j_xav = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode interfacial current density"
        ]

        if self.domain == "Negative":
            self.rhs = {
                c_s_rxav: pybamm.source(-3 * j_xav / self.param.a_R_n, c_s_rxav)
            }

        elif self.domain == "Positive":
            self.rhs = {
                c_s_rxav: pybamm.source(
                    -3 * j_xav / self.param.a_R_p / self.param.gamma_p, c_s_rxav
                )
            }

        if self.name == "quartic profile":
            # We solve an extra ODE for the average particle concentration gradient
            q_s_rxav = variables[
                "Average " + self.domain.lower() + " particle concentration gradient"
            ]
            c_s_surf_xav = variables[
                "X-averaged " + self.domain.lower() + " particle surface concentration"
            ]
            T_xav = variables[
                "X-averaged " + self.domain.lower() + " electrode temperature"
            ]
            if self.domain == "Negative":
                self.rhs.update(
                    {
                        q_s_rxav: pybamm.source(
                            -30
                            * self.param.D_n(c_s_surf_xav, T_xav)
                            * q_s_rxav
                            / self.param.C_n
                            - 45 * j_xav / self.param.a_R_n / 2,
                            q_s_rxav,
                        )
                    }
                )
            elif self.domain == "Positive":
                self.rhs.update(
                    {
                        q_s_rxav: pybamm.source(
                            -30
                            * self.param.D_p(c_s_surf_xav, T_xav)
                            * q_s_rxav
                            / self.param.C_p
                            - 45 * j_xav / self.param.a_R_p / self.param.gamma_p / 2,
                            q_s_rxav,
                        )
                    }
                )

    def set_initial_conditions(self, variables):
        """
        For single particle models, initial conditions can't depend on x so we
        arbitrarily evaluate them at x=0 in the negative electrode and x=1 in the
        positive electrode (they will usually be constant)
        """
        c_s_rxav = variables[
            "Average " + self.domain.lower() + " particle concentration"
        ]

        if self.domain == "Negative":
            c_init = self.param.c_n_init(0)

        elif self.domain == "Positive":
            c_init = self.param.c_p_init(1)

        self.initial_conditions = {c_s_rxav: c_init}
        if self.name == "quartic profile":
            # We also need to provide an initial condition for the average
            # concentration gradient
            q_s_rxav = variables[
                "Average " + self.domain.lower() + " particle concentration gradient"
            ]
            self.initial_conditions.update({q_s_rxav: 0})
