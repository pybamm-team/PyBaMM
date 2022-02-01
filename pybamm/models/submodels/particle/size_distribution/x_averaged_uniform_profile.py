#
# Class for a x-averaged particle-size distribution, with uniform concentration
# profile in each particle
#
import pybamm

from .base_distribution import BaseSizeDistribution


class XAveragedUniformProfile(BaseSizeDistribution):
    """Class for molar conservation in an x-averaged particle-size
    distribution with uniform concentration in each particle.

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
        pybamm.citations.register("Kirk2020")

    def get_fundamental_variables(self):
        # The concentration is uniform throughout each particle, so we
        # can just use the surface value.

        if self.domain == "Negative":
            c_s_surf_xav_distribution = pybamm.Variable(
                "X-averaged negative particle surface concentration distribution",
                domain="negative particle size",
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, 1),
            )
            # Since concentration does not depend on "x", need a particle-size
            # spatial variable R with only "current collector" as secondary
            # domain
            R = pybamm.SpatialVariable(
                "R_n",
                domain=["negative particle size"],
                auxiliary_domains={"secondary": "current collector"},
                coord_sys="cartesian",
            )

        elif self.domain == "Positive":
            c_s_surf_xav_distribution = pybamm.Variable(
                "X-averaged positive particle surface concentration distribution",
                domain="positive particle size",
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, 1),
            )
            # Since concentration does not depend on "x", need a particle-size
            # spatial variable R with only "current collector" as secondary
            # domain
            R = pybamm.SpatialVariable(
                "R_p",
                domain=["positive particle size"],
                auxiliary_domains={"secondary": "current collector"},
                coord_sys="cartesian",
            )

        variables = self._get_distribution_variables(R)

        # Standard distribution variables (size-dependent)
        variables.update(
            self._get_standard_concentration_distribution_variables(
                c_s_surf_xav_distribution
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
            "X-averaged "
            + self.domain.lower()
            + " volume-weighted particle-size distribution"
        ]
        c_s_surf_xav = pybamm.Integral(f_v_dist * c_s_surf_xav_distribution, R)
        c_s_xav = pybamm.PrimaryBroadcast(
            c_s_surf_xav, [self.domain.lower() + " particle"]
        )
        c_s = pybamm.SecondaryBroadcast(c_s_xav, [self.domain.lower() + " electrode"])
        variables.update(self._get_standard_concentration_variables(c_s, c_s_xav))

        # Size-averaged flux variables
        N_s_xav = pybamm.FullBroadcastToEdges(
            0, self.domain.lower() + " particle", "current collector"
        )
        variables.update(self._get_standard_flux_variables(N_s_xav, N_s_xav))
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_total_concentration_variables(variables))
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
        R = variables[self.domain + " particle sizes"]

        if self.domain == "Negative":
            self.rhs = {
                c_s_surf_xav_distribution: -3
                * j_xav_distribution
                / self.param.a_R_n
                / self.param.gamma_n
                / R
            }

        elif self.domain == "Positive":
            self.rhs = {
                c_s_surf_xav_distribution: -3
                * j_xav_distribution
                / self.param.a_R_p
                / self.param.gamma_p
                / R
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
