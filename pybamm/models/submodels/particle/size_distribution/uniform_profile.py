#
# Class for particle-size distributions, one distribution at every
# x location of the electrode, with uniform concentration in each
# particle
#
import pybamm

from .base_distribution import BaseSizeDistribution


class UniformProfile(BaseSizeDistribution):
    """
    Class for molar conservation in particle-size distributions, one
    distribution at every x location of the electrode,
    with a uniform concentration within each particle (in r). Concentration varies
    with R (particle size), and x (electrode coordinate).

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
        pybamm.citations.register("Kirk2021")

    def get_fundamental_variables(self):
        # The concentration is uniform throughout each particle, so we
        # can just use the surface value.

        # distribution variables
        c_s_surf_distribution = pybamm.Variable(
            f"{self.domain} particle surface concentration distribution",
            domain=f"{self.domain.lower()} particle size",
            auxiliary_domains={
                "secondary": f"{self.domain.lower()} electrode",
                "tertiary": "current collector",
            },
            bounds=(0, 1),
        )
        if self.domain == "Negative":
            R = pybamm.standard_spatial_vars.R_n
        elif self.domain == "Positive":
            R = pybamm.standard_spatial_vars.R_p

        variables = self._get_distribution_variables(R)

        # Standard concentration distribution variables (size-dependent)
        variables.update(
            self._get_standard_concentration_distribution_variables(
                c_s_surf_distribution
            )
        )
        # Flux variables (size-dependent)
        variables.update(
            self._get_standard_flux_distribution_variables(pybamm.Scalar(0))
        )

        # Standard size-averaged variables. Average concentrations using
        # the volume-weighted distribution since they are volume-based
        # quantities. Necessary for output variables "Total lithium in
        # negative electrode [mol]", etc, to be calculated correctly
        f_v_dist = variables[
            self.domain + " volume-weighted particle-size distribution"
        ]
        c_s_surf = pybamm.Integral(f_v_dist * c_s_surf_distribution, R)
        c_s = pybamm.PrimaryBroadcast(c_s_surf, [self.domain.lower() + " particle"])
        c_s_xav = pybamm.x_average(c_s)
        variables.update(self._get_standard_concentration_variables(c_s, c_s_xav))

        # Size-averaged flux variables
        N_s = pybamm.FullBroadcastToEdges(
            0,
            [self.domain.lower() + " particle"],
            auxiliary_domains={
                "secondary": self.domain.lower() + " electrode",
                "tertiary": "current collector",
            },
        )
        N_s_xav = pybamm.FullBroadcastToEdges(
            0, self.domain.lower() + " particle", "current collector"
        )
        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_total_concentration_variables(variables))
        return variables

    def set_rhs(self, variables):
        c_s_surf_distribution = variables[
            self.domain + " particle surface concentration distribution"
        ]
        j_distribution = variables[
            self.domain + " electrode interfacial current density distribution"
        ]
        R = variables[self.domain + " particle sizes"]

        self.rhs = {
            c_s_surf_distribution: -3
            * j_distribution
            / self.domain_param.a_R
            / self.domain_param.gamma
            / R
        }

    def set_initial_conditions(self, variables):
        c_s_surf_distribution = variables[
            f"{self.domain} particle surface concentration distribution"
        ]

        c_init = pybamm.PrimaryBroadcast(
            pybamm.r_average(self.domain_param.c_init),
            f"{self.domain.lower()} particle size",
        )

        self.initial_conditions = {c_s_surf_distribution: c_init}
