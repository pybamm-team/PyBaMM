#
# Class for single particle with polynomial concentration profile
#
import pybamm

from .base_particle import BaseParticle


class PolynomialSingleParticle(BaseParticle):
    """
    Base class for molar conservation in a single x-averaged particle with
    an assumed polynomial concentration profile in r.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    order : int, optional
        The order of the polynomial, can be 0 or 2. Default is 2.


    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param, domain, order=2):
        super().__init__(param, domain)
        self.order = order

    def get_fundamental_variables(self):
        if self.domain == "Negative":
            c_s_rxav = pybamm.standard_variables.c_s_n_rxav

        elif self.domain == "Positive":
            c_s_rxav = pybamm.standard_variables.c_s_p_rxav

        variables = {
            "R-X-averaged " + self.domain.lower() + " particle concentration": c_s_rxav
        }

        return variables

    def get_coupled_variables(self, variables):
        c_s_rxav = variables[
            "R-X-averaged " + self.domain.lower() + " particle concentration"
        ]
        i_boundary_cc = variables["Current collector current density"]
        T_xav = pybamm.PrimaryBroadcast(
            variables["X-averaged " + self.domain.lower() + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )

        # Set surface concentration based on polynomial order
        if self.order == 0:
            # The concentration is uniform so the surface value is equal to
            # the average
            c_s_surf_xav = c_s_rxav
        elif self.order == 2:
            # The surface concentration is computed from the average concentration
            # and boundary flux
            # Note 1: here we use the total average interfacial current for the single
            # particle. We explicitly write this as the current density divided by the
            # electrode thickness instead of getting the average current from the
            # interface submodel since the interface requires the surface concentration
            # to be defined to compute the exchange current density. Explicitly writing
            # out the average interfacial current here avoids KeyErrors where variables
            # have not been set in the right order.
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
                    / self.param.a_n
                    / self.param.D_n(c_s_rxav, pybamm.surf(T_xav))
                )

            if self.domain == "Positive":
                j_xav = -i_boundary_cc / self.param.l_p
                c_s_surf_xav = c_s_rxav - self.param.C_p * (
                    j_xav
                    / 5
                    / self.param.a_p
                    / self.param.gamma_p
                    / self.param.D_p(c_s_rxav, pybamm.surf(T_xav))
                )

        # Set concentration depending on polynomial order
        if self.order == 0:
            # The concentration is uniform
            c_s_xav = pybamm.PrimaryBroadcast(
                c_s_rxav, [self.domain.lower() + " particle"]
            )
        elif self.order == 2:
            # The concentration is given by c = A + B*r**2
            A = pybamm.PrimaryBroadcast(
                (1 / 2) * (5 * c_s_rxav - 3 * c_s_surf_xav),
                [self._domain.lower() + " particle"],
            )
            B = pybamm.PrimaryBroadcast(
                (5 / 2) * (c_s_surf_xav - c_s_rxav), [self.domain.lower() + " particle"]
            )
            if self.domain == "Negative":
                # TODO: figure out how to just use r here without getting shape errors
                # r = pybamm.standard_spatial_vars.r_n
                r = pybamm.SpatialVariable(
                    "r_n",
                    domain=["negative particle"],
                    auxiliary_domains={"secondary": "current collector"},
                    coord_sys="spherical polar",
                )
                c_s_xav = A + B * r ** 2
            if self.domain == "Positive":
                # TODO: figure out how to just use r here without getting shape errors
                # r = pybamm.standard_spatial_vars.r_p
                r = pybamm.SpatialVariable(
                    "r_p",
                    domain=["positive particle"],
                    auxiliary_domains={"secondary": "current collector"},
                    coord_sys="spherical polar",
                )
                c_s_xav = A + B * r ** 2

        c_s = pybamm.SecondaryBroadcast(c_s_xav, [self.domain.lower() + " electrode"])

        # Set flux based on polynomial order
        if self.order == 0:
            # The flux is zero since there is no concentration gradient
            N_s_xav = pybamm.FullBroadcastToEdges(
                0, self.domain.lower() + " particle", "current collector"
            )
        elif self.order == 2:
            # The flux may be computed directly from the polynomial for c
            if self.domain == "Negative":
                N_s_xav = (
                    -self.param.D_n(c_s_xav, T_xav) * 5 * (c_s_surf_xav - c_s_rxav) * r
                )
            if self.domain == "Positive":
                N_s_xav = (
                    -self.param.D_p(c_s_xav, T_xav) * 5 * (c_s_surf_xav - c_s_rxav) * r
                )

        N_s = pybamm.SecondaryBroadcast(N_s_xav, [self._domain.lower() + " electrode"])

        variables = self._get_standard_concentration_variables(c_s, c_s_av=c_s_rxav)
        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))

        return variables

    def set_rhs(self, variables):

        c_s_rxav = variables[
            "R-X-averaged " + self.domain.lower() + " particle concentration"
        ]
        j_xav = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode interfacial current density"
        ]

        if self.domain == "Negative":
            self.rhs = {c_s_rxav: -3 * j_xav / self.param.a_n}

        elif self.domain == "Positive":
            self.rhs = {c_s_rxav: -3 * j_xav / self.param.a_p / self.param.gamma_p}

    def set_initial_conditions(self, variables):
        """
        For single particle models, initial conditions can't depend on x so we
        arbitrarily evaluate them at x=0 in the negative electrode and x=1 in the
        positive electrode (they will usually be constant)
        """
        c_s_rxav = variables[
            "R-X-averaged " + self.domain.lower() + " particle concentration"
        ]

        if self.domain == "Negative":
            c_init = self.param.c_n_init(0)

        elif self.domain == "Positive":
            c_init = self.param.c_p_init(1)

        self.initial_conditions = {c_s_rxav: c_init}
