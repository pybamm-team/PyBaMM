#
# Class for particle-size distributions, one distribution at every
# x location of the electrode, and Fickian diffusion within each particle
#
import pybamm

from .base_distribution import BaseSizeDistribution


class FickianDiffusion(BaseSizeDistribution):
    """Class for molar conservation in particle-size distributions, one
    distribution at every x location of the electrode,
    with Fickian diffusion within each particle. Concentration varies with
    r (spherical coordinate), R (particle size), and x (electrode coordinate).

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
        if self.domain == "Negative":
            c_s_distribution = pybamm.Variable(
                "Negative particle concentration distribution",
                domain="negative particle",
                auxiliary_domains={
                    "secondary": "negative particle size",
                    "tertiary": "negative electrode",
                    "quaternary": "current collector",
                },
                bounds=(0, 1),
            )
            R = pybamm.standard_spatial_vars.R_n

        elif self.domain == "Positive":
            c_s_distribution = pybamm.Variable(
                "Positive particle concentration distribution",
                domain="positive particle",
                auxiliary_domains={
                    "secondary": "positive particle size",
                    "tertiary": "positive electrode",
                    "quaternary": "current collector",
                },
                bounds=(0, 1),
            )
            R = pybamm.standard_spatial_vars.R_p

        variables = self._get_distribution_variables(R)

        # Standard concentration distribution variables (size-dependent)
        variables.update(
            self._get_standard_concentration_distribution_variables(c_s_distribution)
        )

        # Standard size-averaged variables. Average concentrations using
        # the volume-weighted distribution since they are volume-based
        # quantities. Necessary for output variables "Total lithium in
        # negative electrode [mol]", etc, to be calculated correctly
        f_v_dist = variables[
            self.domain + " volume-weighted particle-size distribution"
        ]
        c_s = pybamm.Integral(f_v_dist * c_s_distribution, R)
        c_s_xav = pybamm.x_average(c_s)
        variables.update(self._get_standard_concentration_variables(c_s, c_s_xav))

        return variables

    def get_coupled_variables(self, variables):
        c_s_distribution = variables[
            self.domain + " particle concentration distribution"
        ]
        R = variables[self.domain + " particle sizes"]

        # broadcast T to "particle size" domain then again into "particle"
        T_k = pybamm.PrimaryBroadcast(
            variables[self.domain + " electrode temperature"],
            [self.domain.lower() + " particle size"],
        )
        T_k = pybamm.PrimaryBroadcast(
            T_k,
            [self.domain.lower() + " particle"],
        )

        if self.domain == "Negative":
            N_s_distribution = -self.param.D_n(c_s_distribution, T_k) * pybamm.grad(
                c_s_distribution
            )
            f_a_dist = self.param.f_a_dist_n(R)
        elif self.domain == "Positive":
            N_s_distribution = -self.param.D_p(c_s_distribution, T_k) * pybamm.grad(
                c_s_distribution
            )
            f_a_dist = self.param.f_a_dist_p(R)

        # Size-dependent flux variables
        variables.update(
            self._get_standard_flux_distribution_variables(N_s_distribution)
        )

        # Size-averaged flux variables (perform area-weighted avg manually as flux
        # evals on edges)
        N_s = pybamm.Integral(f_a_dist * N_s_distribution, R)
        variables.update(self._get_standard_flux_variables(N_s, N_s))

        variables.update(self._get_total_concentration_variables(variables))
        return variables

    def set_rhs(self, variables):
        c_s_distribution = variables[
            self.domain + " particle concentration distribution"
        ]
        N_s_distribution = variables[self.domain + " particle flux distribution"]
        R = pybamm.PrimaryBroadcast(
            variables[self.domain + " particle sizes"],
            [self.domain.lower() + " particle"],
        )

        if self.domain == "Negative":
            self.rhs = {
                c_s_distribution: -(1 / self.param.C_n)
                * pybamm.div(N_s_distribution)
                / R ** 2
            }
        elif self.domain == "Positive":
            self.rhs = {
                c_s_distribution: -(1 / self.param.C_p)
                * pybamm.div(N_s_distribution)
                / R ** 2
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
        R = variables[self.domain + " particle sizes"]
        T_k = pybamm.PrimaryBroadcast(
            variables[self.domain + " electrode temperature"],
            [self.domain.lower() + " particle size"],
        )

        # Set surface Neumann boundary values
        if self.domain == "Negative":
            rbc = (
                -self.param.C_n
                * R
                * j_distribution
                / self.param.a_R_n
                / self.param.gamma_n
                / self.param.D_n(c_s_surf_distribution, T_k)
            )
        elif self.domain == "Positive":
            rbc = (
                -self.param.C_p
                * R
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
            c_init = pybamm.SecondaryBroadcast(
                self.param.c_n_init, "negative particle size"
            )
        elif self.domain == "Positive":
            c_init = pybamm.SecondaryBroadcast(
                self.param.c_p_init, "positive particle size"
            )

        self.initial_conditions = {c_s_distribution: c_init}
