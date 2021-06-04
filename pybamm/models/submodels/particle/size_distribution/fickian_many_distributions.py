#
# Class for many particle-size distributions, one distribution at every
# x location of the electrode, and Fickian diffusion within each particle
#
import pybamm

from .base_distribution import BaseSizeDistribution


class FickianManySizeDistributions(BaseSizeDistribution):
    """Class for molar conservation in many particle-size
    distributions, one distribution at every x location of the electrode,
    with Fickian diffusion within each particle.

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
        if self.domain == "Negative":
            # distribution variables
            c_s_distribution = pybamm.Variable(
                "Negative particle concentration distribution",
                domain="negative particle",
                auxiliary_domains={
                    "secondary": "negative particle size",
                    "tertiary": "negative electrode",
                },
                bounds=(0, 1),
            )
            # Since concentration does not depend on "y,z", need a particle-size
            # spatial variable R with only "electrode" as secondary
            # domain
            R_variable = pybamm.SpatialVariable(
                "R_n",
                domain=["negative particle size"],
                auxiliary_domains={
                    "secondary": "negative electrode",
                },
                coord_sys="cartesian",
            )
            #R_variable = pybamm.standard_spatial_vars.R_n
            R_dim = self.param.R_n_typ

            # Particle-size distribution (area-weighted)
            f_a_dist = self.param.f_a_dist_n(R_variable)

        elif self.domain == "Positive":
            # distribution variables
            c_s_distribution = pybamm.Variable(
                "Positive particle concentration distribution",
                domain="positive particle",
                auxiliary_domains={
                    "secondary": "positive particle size",
                    "tertiary": "positive electrode",
                },
                bounds=(0, 1),
            )
            # Since concentration does not depend on "y,z", need a
            # spatial variable R with only "electrode" as secondary
            # domain
            R_variable = pybamm.SpatialVariable(
                "R_p",
                domain=["positive particle size"],
                auxiliary_domains={
                    "secondary": "positive electrode",
                },
                coord_sys="cartesian",
            )
            #R = pybamm.standard_spatial_vars.R_p  # used for averaging
            #R_variable = pybamm.SecondaryBroadcast(R, ["positive electrode"])
            R_dim = self.param.R_p_typ

            # Particle-size distribution (area-weighted)
            f_a_dist = self.param.f_a_dist_p(R_variable)

        # Ensure the distribution is normalised, irrespective of discretisation
        # or user input
        f_a_dist = f_a_dist / pybamm.Integral(f_a_dist, R_variable)

        # Standard R-averaged variables (avg secondary domain)
        c_s = pybamm.Integral(f_a_dist * c_s_distribution, R_variable)
        c_s_xav = pybamm.x_average(c_s)
        variables = self._get_standard_concentration_variables(c_s, c_s_xav)

        # Standard distribution variables (R-dependent)
        variables.update(
            self._get_standard_concentration_distribution_variables(c_s_distribution)
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

    def get_coupled_variables(self, variables):
        c_s_distribution = variables[
            self.domain + " particle concentration distribution"
        ]
        R_variable = variables[self.domain + " particle size"]
        R = pybamm.PrimaryBroadcast(R_variable, [self.domain.lower() + " particle"],)
        T_k = variables[self.domain + " electrode temperature"]

        # Variables can currently only have 3 domains, so remove "current collector"
        # from T_k. If T_k was broadcast to "electrode", take orphan, average
        # over "current collector", then broadcast to "particle", "particle-size"
        # and "electrode"
        if isinstance(T_k, pybamm.Broadcast):
            T_k = pybamm.yz_average(T_k.orphans[0])
            T_k = pybamm.FullBroadcast(
                T_k, self.domain.lower() + " particle",
                {
                    "secondary": self.domain.lower() + " particle size",
                    "tertiary": self.domain.lower() + " electrode"
                }
            )
        else:
            # broadcast to "particle size" domain then again into "particle"
            T_k = pybamm.PrimaryBroadcast(
                T_k,
                [self.domain.lower() + " particle size"],
            )
            T_k = pybamm.PrimaryBroadcast(
                T_k, [self.domain.lower() + " particle"],
            )

        if self.domain == "Negative":
            N_s_distribution = (
                -self.param.D_n(c_s_distribution, T_k)
                * pybamm.grad(c_s_distribution)
                / R
            )
            f_a_dist = self.param.f_a_dist_n(R_variable)

            # spatial var to use in R integral below (cannot use R_variable as
            # it is a broadcast)
            #R = pybamm.standard_spatial_vars.R_n
        elif self.domain == "Positive":
            N_s_distribution = (
                -self.param.D_p(c_s_distribution, T_k)
                * pybamm.grad(c_s_distribution)
                / R
            )
            f_a_dist = self.param.f_a_dist_p(R_variable)

            # spatial var to use in R integral below (cannot use R_variable as
            # it is a broadcast)
            #R = pybamm.standard_spatial_vars.R_p

        # Standard R-averaged flux variables
        N_s = pybamm.Integral(f_a_dist * N_s_distribution, R_variable)
        variables.update(self._get_standard_flux_variables(N_s, N_s))

        # Standard distribution flux variables (R-dependent)
        variables.update(
            {self.domain + " particle flux distribution": N_s_distribution}
        )
        return variables

    def set_rhs(self, variables):
        c_s_distribution = variables[
            self.domain + " particle concentration distribution"
        ]

        N_s_distribution = variables[self.domain + " particle flux distribution"]

        R_variable = variables[self.domain + " particle size"]
        R = pybamm.PrimaryBroadcast(R_variable, [self.domain.lower() + " particle"],)

        if self.domain == "Negative":
            self.rhs = {
                c_s_distribution: -(1 / self.param.C_n)
                * pybamm.div(N_s_distribution)
                / R
            }
        elif self.domain == "Positive":
            self.rhs = {
                c_s_distribution: -(1 / self.param.C_p)
                * pybamm.div(N_s_distribution)
                / R
            }

    def set_boundary_conditions(self, variables):
        # Extract variables
        c_s_distribution = variables[
            self.domain + " particle concentration distribution"
        ]
        c_s_surf_distribution = variables[
            self.domain + " particle surface concentration distribution"
        ]
        j_distribution = variables[
            self.domain + " electrode interfacial current density distribution"
        ]
        R_variable = variables[self.domain + " particle size"]

        # Extract T and broadcast to particle size domain
        T_k = variables[self.domain + " electrode temperature"]
        T_k = pybamm.PrimaryBroadcast(
            T_k, [self.domain.lower() + " particle size"]
        )

        # Set surface Neumann boundary values
        if self.domain == "Negative":
            rbc = (
                -self.param.C_n
                * R_variable
                * j_distribution
                / self.param.a_R_n
                / self.param.D_n(c_s_surf_distribution, T_k)
            )

        elif self.domain == "Positive":
            rbc = (
                -self.param.C_p
                * R_variable
                * j_distribution
                / self.param.a_R_p
                / self.param.gamma_p
                / self.param.D_p(c_s_surf_distribution, T_k)
            )

        self.boundary_conditions = {
            c_s_distribution: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (rbc, "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):
        c_s_distribution = variables[
            self.domain + " particle concentration distribution"
        ]

        if self.domain == "Negative":
            # Broadcast x_n to particle-size then into the particles
            x_n = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_n, "negative particle size"
            )
            x_n = pybamm.PrimaryBroadcast(x_n, "negative particle")
            c_init = self.param.c_n_init(x_n)

        elif self.domain == "Positive":
            # Broadcast x_n to particle-size then into the particles
            x_p = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_p, "positive particle size"
            )
            x_p = pybamm.PrimaryBroadcast(x_p, "positive particle")
            c_init = self.param.c_p_init(x_p)

        self.initial_conditions = {c_s_distribution: c_init}

    def set_events(self, variables):
        c_s_surf_distribution = variables[
            self.domain + " particle surface concentration distribution"
        ]
        tol = 1e-5

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
