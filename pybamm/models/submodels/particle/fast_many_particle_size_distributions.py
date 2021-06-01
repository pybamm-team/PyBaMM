#
# Class for many particle-size distributions, one distribution at every
# x location of the electrode, with fast diffusion (uniform concentration in r)
# within particles
#
import pybamm

from .base_particle import BaseParticle


class FastManyPSDs(BaseParticle):
    """Class for molar conservation in many particle-size
    distributions (PSD), one distribution at every x location of the electrode,
    with fast diffusion (uniform concentration in r) within the particles

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.particle.BaseParticle`
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
            c_s_surf_distribution = pybamm.Variable(
                "Negative particle surface concentration distribution",
                domain="negative particle-size domain",
                auxiliary_domains={
                    "secondary": "negative electrode",
                    "tertiary": "current collector",
                },
                bounds=(0, 1),
            )
            R = pybamm.standard_spatial_vars.R_variable_n  # used for averaging
            R_variable = pybamm.SecondaryBroadcast(R, ["current collector"])
            R_variable = pybamm.SecondaryBroadcast(R_variable, ["negative electrode"])
            R_dim = self.param.R_n

            # Particle-size distribution (area-weighted)
            f_a_dist = self.param.f_a_dist_n(R_variable)

        elif self.domain == "Positive":
            # distribution variables
            c_s_surf_distribution = pybamm.Variable(
                "Positive particle surface concentration distribution",
                domain="positive particle-size domain",
                auxiliary_domains={
                    "secondary": "positive electrode",
                    "tertiary": "current collector",
                },
                bounds=(0, 1),
            )
            R = pybamm.standard_spatial_vars.R_variable_p  # used for averaging
            R_variable = pybamm.SecondaryBroadcast(R, ["current collector"])
            R_variable = pybamm.SecondaryBroadcast(R_variable, ["positive electrode"])
            R_dim = self.param.R_p

            # Particle-size distribution (area-weighted)
            f_a_dist = self.param.f_a_dist_p(R_variable)

        # Ensure the distribution is normalised, irrespective of discretisation
        # or user input
        f_a_dist = f_a_dist / pybamm.Integral(f_a_dist, R)

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
        c_s_surf = pybamm.Integral(f_a_dist * c_s_surf_distribution, R)
        c_s = pybamm.PrimaryBroadcast(
            c_s_surf, [self.domain.lower() + " particle"]
        )
        c_s_xav = pybamm.x_average(c_s)
        variables = self._get_standard_concentration_variables(c_s, c_s_xav)
        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))

        # Standard distribution variables (R-dependent)
        variables.update(
            self._get_standard_concentration_distribution_variables(
                c_s_surf_distribution
            )
        )
        variables.update(
            {
                self.domain + " particle size": R_variable,
                self.domain + " particle size [m]": R_variable * R_dim,
                self.domain
                + " area-weighted particle-size"
                + " distribution": pybamm.x_average(f_a_dist),
                self.domain
                + " area-weighted particle-size"
                + " distribution [m-1]": pybamm.x_average(f_a_dist) / R_dim,
            }
        )
        return variables

    def set_rhs(self, variables):
        c_s_surf_distribution = variables[
            self.domain
            + " particle surface concentration distribution"
        ]
        j_distribution = variables[
            self.domain
            + " electrode interfacial current density distribution"
        ]
        R_variable = variables[self.domain + " particle size"]

        if self.domain == "Negative":
            self.rhs = {
                c_s_surf_distribution: -3
                * j_distribution
                / self.param.a_R_n
                / R_variable
            }

        elif self.domain == "Positive":
            self.rhs = {
                c_s_surf_distribution: -3
                * j_distribution
                / self.param.a_R_p
                / self.param.gamma_p
                / R_variable
            }

    def set_initial_conditions(self, variables):
        c_s_surf_distribution = variables[
            self.domain
            + " particle surface concentration distribution"
        ]

        if self.domain == "Negative":
            # Broadcast x_n to particle-size domain
            x_n = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_n, "negative particle-size domain"
            )
            c_init = self.param.c_n_init(x_n)

        elif self.domain == "Positive":
            # Broadcast x_p to particle-size domain
            x_p = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_p, "positive particle-size domain"
            )
            c_init = self.param.c_p_init(x_p)

        self.initial_conditions = {c_s_surf_distribution: c_init}

    def set_events(self, variables):
        c_s_surf_distribution = variables[
            self.domain
            + " particle surface concentration distribution"
        ]
        tol = 1e-4

        self.events.append(
            pybamm.Event(
                "Minumum " + self.domain.lower() + " particle surface concentration",
                pybamm.min(c_s_surf_distribution) - tol,
                pybamm.EventType.TERMINATION,
            )
        )

        self.events.append(
            pybamm.Event(
                "Maximum " + self.domain.lower() + " particle surface concentration",
                (1 - tol) - pybamm.max(c_s_surf_distribution),
                pybamm.EventType.TERMINATION,
            )
        )
