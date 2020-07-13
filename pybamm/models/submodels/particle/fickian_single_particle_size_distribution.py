#
# Class for a single particle with Fickian diffusion
#
import pybamm

from .base_particle import BaseParticle


class FickianSinglePSD(BaseParticle):
    """Base class for molar conservation in a single x-averaged particle which employs
    Fick's law.

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

    def get_fundamental_variables(self):
        if self.domain == "Negative":
            # distribution variables
            c_s_xav_distribution = pybamm.Variable(
                "X-averaged negative particle concentration distribution",
                domain="negative particle",
                auxiliary_domains={
                    "secondary": "negative particle-size domain",
                    "tertiary": "current collector",
                },
                bounds=(0, 1),
            )
            j_xav_distribution = pybamm.Variable(
                "X-averaged negative electrode interfacial current density",
                domain="negative particle-size domain",
                auxiliary_domains={"secondary": "current collector",},
            )
            R_variable = pybamm.standard_spatial_vars.R_variable_n

            # Additional parameters for this model
            # Dimensionless standard deviation
            sd_a = self.param.sd_a_n

            # Particle-size distribution (area-weighted)
            f_a_dist = self.param.f_a_dist_n(R_variable, 1, sd_a)

        elif self.domain == "Positive":
            # distribution variables
            c_s_xav_distribution = pybamm.Variable(
                "X-averaged positive particle concentration distribution",
                domain="positive particle",
                auxiliary_domains={
                    "secondary": "positive particle-size domain",
                    "tertiary": "current collector",
                },
                bounds=(0, 1),
            )
            j_xav_distribution = pybamm.Variable(
                "X-averaged positive electrode interfacial current density",
                domain="positive particle-size domain",
                auxiliary_domains={"secondary": "current collector"},
            )
            R_variable = pybamm.standard_spatial_vars.R_variable_p

            # Additional parameters for this model
            # Dimensionless standard deviation
            sd_a = self.param.sd_a_p

            # Particle-size distribution (area-weighted)
            f_a_dist = self.param.f_a_dist_p(R_variable, 1, sd_a)

        # R-averaged variables
        c_s_xav = pybamm.Integral(f_a_dist * c_s_xav_distribution, R_variable)
        c_s = pybamm.SecondaryBroadcast(c_s_xav, [self.domain.lower() + " electrode"])

        c_s_surf_xav_distribution = pybamm.surf(c_s_xav_distribution)
        c_s_surf_distribution = pybamm.SecondaryBroadcast(
            c_s_surf_xav_distribution, [self.domain.lower() + " electrode"]
        )
        variables = self._get_standard_concentration_variables(c_s, c_s_xav)
        variables.update(
            {
                self.domain + " particle size": R_variable,
                "X-averaged "
                + self.domain.lower()
                + " particle concentration distribution": c_s_xav_distribution,
                "X-averaged "
                + self.domain.lower()
                + " particle surface concentration"
                + " distribution": c_s_surf_xav_distribution,
                self.domain
                + " particle surface concentration"
                + " distribution": c_s_surf_distribution,
                self.domain
                + " area-weighted particle-size"
                + " distribution": f_a_dist,
            }
        )
        return variables

    def get_coupled_variables(self, variables):
        c_s_xav_distribution = variables[
            "X-averaged " + self.domain.lower() + " particle concentration distribution"
        ]
        R_variable = variables[self.domain + " particle size"]

        # broadcast to particle-size domain
        T_k_xav = pybamm.PrimaryBroadcast(
            variables["X-averaged " + self.domain.lower() + " electrode temperature"],
            [self.domain.lower() + " particle-size domain"],
        )

        # broadcast again into particle
        T_k_xav = pybamm.PrimaryBroadcast(T_k_xav, [self.domain.lower() + " particle"],)

        if self.domain == "Negative":
            N_s_xav_distribution = -self.param.D_n(
                c_s_xav_distribution, T_k_xav
            ) * pybamm.grad(c_s_xav_distribution)
            f_a_dist = variables["Negative area-weighted particle-size distribution"]
            N_s_xav = pybamm.Integral(f_a_dist * N_s_xav_distribution, R_variable)
        elif self.domain == "Positive":
            N_s_xav_distribution = -self.param.D_p(
                c_s_xav_distribution, T_k_xav
            ) * pybamm.grad(c_s_xav_distribution)
            f_a_dist = variables["Positive area-weighted particle-size distribution"]
            N_s_xav = pybamm.Integral(f_a_dist * N_s_xav_distribution, R_variable)

        N_s = pybamm.SecondaryBroadcast(N_s_xav, [self._domain.lower() + " electrode"])

        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))
        variables.update(
            {
                "X-averaged "
                + self.domain.lower()
                + " particle flux distribution": N_s_xav_distribution,
            }
        )
        return variables

    def set_rhs(self, variables):
        c_s_xav_distribution = variables[
            "X-averaged " + self.domain.lower() + " particle concentration distribution"
        ]

        N_s_xav_distribution = variables[
            "X-averaged " + self.domain.lower() + " particle flux distribution"
        ]

        R_variable = variables[self.domain + " particle size"]
        if self.domain == "Negative":
            self.rhs = {
                c_s_xav_distribution: -(1 / self.param.C_n)
                * pybamm.div(N_s_xav_distribution)
                / R_variable ** 2
            }
        elif self.domain == "Positive":
            self.rhs = {
                c_s_xav_distribution: -(1 / self.param.C_p)
                * pybamm.div(N_s_xav_distribution)
                / R_variable ** 2
            }

    def set_boundary_conditions(self, variables):
        c_s_xav_distribution = variables[
            "X-averaged " + self.domain.lower() + " particle concentration distribution"
        ]

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

        T_k_xav = variables[
            "X-averaged " + self.domain.lower() + " electrode temperature"
        ]
        T_k_xav = pybamm.PrimaryBroadcast(
            T_k_xav, [self.domain.lower() + " particle-size domain"]
        )

        R_variable = variables[self.domain + " particle size"]

        if self.domain == "Negative":
            rbc = (
                -self.param.C_n
                * R_variable
                * j_xav_distribution
                / self.param.a_n
                / self.param.D_n(c_s_surf_xav_distribution, T_k_xav)
            )

        elif self.domain == "Positive":
            rbc = (
                -self.param.C_p
                * R_variable
                * j_xav_distribution
                / self.param.a_p
                / self.param.gamma_p
                / self.param.D_p(c_s_surf_xav_distribution, T_k_xav)
            )

        self.boundary_conditions = {
            c_s_xav_distribution: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (rbc, "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):
        """
        For single particle-size distribution models, initial conditions can't
        depend on x so we arbitrarily set the initial values of the single
        particles to be given by the values at x=0 in the negative electrode
        and x=1 in the positive electrode. Typically, supplied initial
        conditions are uniform x.
        """
        c_s_xav_distribution = variables[
            "X-averaged " + self.domain.lower() + " particle concentration distribution"
        ]

        if self.domain == "Negative":
            c_init = self.param.c_n_init(0)

        elif self.domain == "Positive":
            c_init = self.param.c_p_init(1)

        self.initial_conditions = {c_s_xav_distribution: c_init}

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
