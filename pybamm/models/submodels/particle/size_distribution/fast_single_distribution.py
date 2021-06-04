#
# Class for a single particle-size distribution representing an
# electrode, with fast diffusion (uniform concentration in r) within particles
#
import pybamm

from .base_distribution import BaseSizeDistribution


class FastSingleSizeDistribution(BaseSizeDistribution):
    """Class for molar conservation in a single (i.e., x-averaged) particle-size
    distribution) with fast diffusion within each particle
    (uniform concentration in r).

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.particle.size_distribution.BaseSizeDistribution`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)
        # pybamm.citations.register("kirk2020")

    def get_fundamental_variables(self):
        # The concentration is uniform throughout each particle, so we
        # can just use the surface value. This avoids dealing with
        # x, R *and* r averaged quantities, which may be confusing.

        if self.domain == "Negative":
            # distribution variables
            c_s_surf_xav_distribution = pybamm.Variable(
                "X-averaged negative particle surface concentration distribution",
                domain="negative particle size",
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, 1),
            )
            R_variable = pybamm.standard_spatial_vars.R_n
            R_dim = self.param.R_n_typ

            # Particle-size distribution (area-weighted)
            f_a_dist = self.param.f_a_dist_n(R_variable)

        elif self.domain == "Positive":
            # distribution variables
            c_s_surf_xav_distribution = pybamm.Variable(
                "X-averaged positive particle surface concentration distribution",
                domain="positive particle size",
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, 1),
            )
            R_variable = pybamm.standard_spatial_vars.R_p
            R_dim = self.param.R_p_typ

            # Particle-size distribution (area-weighted)
            f_a_dist = self.param.f_a_dist_p(R_variable)

        # Ensure the distribution is normalised, irrespective of discretisation
        # or user input
        f_a_dist = f_a_dist / pybamm.Integral(f_a_dist, R_variable)

        # Flux variables (zero)
        N_s = pybamm.FullBroadcastToEdges(
            0,
            [self.domain.lower() + " particle"],
            auxiliary_domains={
                "secondary": self.domain.lower() + " electrode",
                "tertiary": "current collector",
            },
        )
        N_s_xav = pybamm.FullBroadcast(
            0, self.domain.lower() + " electrode", "current collector"
        )

        # Standard R-averaged variables
        c_s_surf_xav = pybamm.Integral(f_a_dist * c_s_surf_xav_distribution, R_variable)
        c_s_xav = pybamm.PrimaryBroadcast(
            c_s_surf_xav, [self.domain.lower() + " particle"]
        )
        c_s = pybamm.SecondaryBroadcast(c_s_xav, [self.domain.lower() + " electrode"])
        variables = self._get_standard_concentration_variables(c_s, c_s_xav)
        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))

        # Standard distribution variables (R-dependent)
        variables.update(
            self._get_standard_concentration_distribution_variables(
                c_s_surf_xav_distribution
            )
        )
        variables.update(
            {
                self.domain + " particle size": R_variable,
                self.domain + " particle size [m]": R_variable * R_dim,
                self.domain
                + " area-weighted particle-size"
                + " distribution": f_a_dist,
                self.domain
                + " area-weighted particle-size"
                + " distribution [m-1]": f_a_dist / R_dim,
            }
        )
        return variables

    def set_rhs(self, variables):
        c_s_surf_xav_distribution = variables[
            "X-averaged "
            + self.domain.lower()
            + " particle surface concentration distribution"
        ]
        j_xav_distribution = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode interfacial current density distribution"
        ]
        R_variable = variables[self.domain + " particle size"]

        if self.domain == "Negative":
            self.rhs = {
                c_s_surf_xav_distribution: -3
                * j_xav_distribution
                / self.param.a_R_n
                / R_variable
            }

        elif self.domain == "Positive":
            self.rhs = {
                c_s_surf_xav_distribution: -3
                * j_xav_distribution
                / self.param.a_R_p
                / self.param.gamma_p
                / R_variable
            }

    def set_initial_conditions(self, variables):
        """
        For single particle-size distribution models, initial conditions can't
        depend on x so we arbitrarily set the initial values of the single
        particles to be given by the values at x=0 in the negative electrode
        and x=1 in the positive electrode. Typically, supplied initial
        conditions are uniform x.
        """
        c_s_surf_xav_distribution = variables[
            "X-averaged "
            + self.domain.lower()
            + " particle surface concentration distribution"
        ]

        if self.domain == "Negative":
            c_init = self.param.c_n_init(0)

        elif self.domain == "Positive":
            c_init = self.param.c_p_init(1)

        self.initial_conditions = {c_s_surf_xav_distribution: c_init}

    def set_events(self, variables):
        c_s_surf_xav_distribution = variables[
            "X-averaged "
            + self.domain.lower()
            + " particle surface concentration distribution"
        ]
        tol = 1e-4

        self.events.append(
            pybamm.Event(
                "Minumum " + self.domain.lower() + " particle surface concentration",
                pybamm.min(c_s_surf_xav_distribution) - tol,
                pybamm.EventType.TERMINATION,
            )
        )

        self.events.append(
            pybamm.Event(
                "Maximum " + self.domain.lower() + " particle surface concentration",
                (1 - tol) - pybamm.max(c_s_surf_xav_distribution),
                pybamm.EventType.TERMINATION,
            )
        )
