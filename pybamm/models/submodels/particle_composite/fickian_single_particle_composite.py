#
# Class for a single particle with Fickian diffusion
#
import pybamm

from .base_particle_composite import BaseParticleComposite


class FickianSingleParticleComposite(BaseParticleComposite):
    """
    Class for molar conservation in a single x-averaged composite particle which employs
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
            c_s_xav = pybamm.standard_variables.c_s_n_xav
            c_s = pybamm.SecondaryBroadcast(c_s_xav, ["negative electrode"])

        elif self.domain == "Positive":
            c_s_xav = pybamm.standard_variables.c_s_p_xav
            c_s = pybamm.SecondaryBroadcast(c_s_xav, ["positive electrode"])

        variables = self._get_standard_concentration_variables(
            c_s, c_s_xav=c_s_xav, phase="phase 1"
        )
        variables.update(
            self._get_standard_concentration_variables(
                c_s, c_s_xav=c_s_xav, phase="phase 2"
            )
        )
        return variables

    def get_coupled_variables(self, variables):
        if self.domain == "Negative":
            p1_name = " of " + self.param.n_p1_name
            p2_name = " of " + self.param.n_p2_name
        elif self.domain == "Positive":
            p1_name = " of " + self.param.p_p1_name
            p2_name = " of " + self.param.p_p2_name

        c_s_xav_p1 = variables[
            f"X-averaged {self.domain.lower()} particle concentration{p1_name}"
        ]
        c_s_xav_p2 = variables[
            f"X-averaged {self.domain.lower()} particle concentration{p2_name}"
        ]
        T_xav = pybamm.PrimaryBroadcast(
            variables["X-averaged " + self.domain.lower() + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )

        if self.domain == "Negative":
            N_s_xav_p1 = -self.param.D_n(c_s_xav_p1, T_xav, "phase 1") * pybamm.grad(
                c_s_xav_p1
            )
            N_s_xav_p2 = -self.param.D_n(c_s_xav_p2, T_xav, "phase 2") * pybamm.grad(
                c_s_xav_p2
            )
        elif self.domain == "Positive":
            N_s_xav_p1 = -self.param.D_p(c_s_xav_p1, T_xav, "phase 1") * pybamm.grad(
                c_s_xav_p1
            )
            N_s_xav_p2 = -self.param.D_p(c_s_xav_p2, T_xav), "phase 2" * pybamm.grad(
                c_s_xav_p2
            )
        N_s_p1 = pybamm.SecondaryBroadcast(
            N_s_xav_p1, [self._domain.lower() + " electrode"]
        )
        N_s_p2 = pybamm.SecondaryBroadcast(
            N_s_xav_p2, [self._domain.lower() + " electrode"]
        )

        variables.update(
            self._get_standard_flux_variables(N_s_p1, N_s_xav_p1, phase="phase 1")
        )
        variables.update(
            self._get_standard_flux_variables(N_s_p2, N_s_xav_p2, phase="phase 2")
        )
        # variables.update(
        #     self._get_standard_flux_variables(N_s_p1 + N_s_p2, N_s_p1_xav + N_s_p2_xav)
        # ) # may not be right to sum them directly
        variables.update(
            self._get_total_concentration_variables(variables, phase="phase 1")
        )
        variables.update(
            self._get_total_concentration_variables(variables, phase="phase 2")
        )
        return variables

    def set_rhs(self, variables):
        if self.domain == "Negative":
            p1_name = " of " + self.param.n_p1_name
            p2_name = " of " + self.param.n_p2_name
            C_s_p1 = self.param.C_n_p1
            C_s_p2 = self.param.C_n_p2
        elif self.domain == "Positive":
            p1_name = " of " + self.param.p_p1_name
            p2_name = " of " + self.param.p_p2_name
            C_s_p1 = self.param.C_p_p1
            C_s_p2 = self.param.C_p_p2
        c_s_xav_p1 = variables[
            f"X-averaged {self.domain.lower()} particle concentration{p1_name}"
        ]
        c_s_xav_p2 = variables[
            f"X-averaged {self.domain.lower()} particle concentration{p2_name}"
        ]
        N_s_xav_p1 = variables[
            f"X-averaged {self.domain.lower()} particle flux{p1_name}"
        ]
        N_s_xav_p2 = variables[
            f"X-averaged {self.domain.lower()} particle flux{p2_name}"
        ]
        self.rhs = {
            c_s_xav_p1: -(1 / C_s_p1) * pybamm.div(N_s_xav_p1),
            c_s_xav_p2: -(1 / C_s_p2) * pybamm.div(N_s_xav_p2),
        }

    def set_boundary_conditions(self, variables):
        if self.domain == "Negative":
            p1_name = " of " + self.param.n_p1_name
            p2_name = " of " + self.param.n_p2_name
        elif self.domain == "Positive":
            p1_name = " of " + self.param.p_p1_name
            p2_name = " of " + self.param.p_p2_name
        c_s_xav_p1 = variables[
            f"X-averaged {self.domain.lower()} particle concentration{p1_name}"
        ]
        c_s_xav_p2 = variables[
            f"X-averaged {self.domain.lower()} particle concentration{p2_name}"
        ]
        c_s_surf_xav_p1 = variables[
            f"X-averaged {self.domain.lower()} particle surface concentration{p1_name}"
        ]
        c_s_surf_xav_p2 = variables[
            f"X-averaged {self.domain.lower()} particle surface concentration{p2_name}"
        ]
        T_xav = variables[
            "X-averaged " + self.domain.lower() + " electrode temperature"
        ]
        j_xav_p1 = variables[
            f"X-averaged {self.domain.lower()}"
            + f" electrode interfacial current density{p1_name}"
        ]
        j_xav_p2 = variables[
            f"X-averaged {self.domain.lower()}"
            + f" electrode interfacial current density{p2_name}"
        ]

        if self.domain == "Negative":
            rbc_b1 = (
                -self.param.C_n_p1
                * j_xav_p1
                / self.param.a_R_n_p1
                / self.param.D_n(c_s_surf_xav_p1, T_xav, "phase 1")
            )
            rbc_b2 = (
                -self.param.C_n_p2
                * j_xav_p2
                / self.param.a_R_n_p2
                / self.param.D_n(c_s_surf_xav_p2, T_xav, "phase 2")
            )

        elif self.domain == "Positive":
            rbc_b1 = (
                -self.param.C_p_p1
                * j_xav_p1
                / self.param.a_R_p_p1
                / self.param.gamma_p
                / self.param.D_p(c_s_surf_xav_p1, T_xav, "phase 1")
            )
            rbc_b2 = (
                -self.param.C_p_p2
                * j_xav_p2
                / self.param.a_R_p_p2
                / self.param.gamma_p
                / self.param.D_p(c_s_surf_xav_p2, T_xav, "phase 2")
            )

        self.boundary_conditions = {
            c_s_xav_p1: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (rbc_p1, "Neumann"),
            },
            c_s_xav_p2: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (rbc_p2, "Neumann"),
            },
        }

    def set_initial_conditions(self, variables):
        """
        For single particle models, initial conditions can't depend on x so we
        arbitrarily set the initial values of the single particles to be given
        by the values at x=0 in the negative electrode and x=1 in the
        positive electrode. Typically, supplied initial conditions are uniform
        x.
        """
        if self.domain == "Negative":
            p1_name = " of " + self.param.n_p1_name
            p2_name = " of " + self.param.n_p2_name
        elif self.domain == "Positive":
            p1_name = " of " + self.param.p_p1_name
            p2_name = " of " + self.param.p_p2_name
        c_s_xav_p1 = variables[
            f"X-averaged {self.domain.lower()} particle concentration{p1_name}"
        ]
        c_s_xav_p2 = variables[
            f"X-averaged {self.domain.lower()} particle concentration{p2_name}"
        ]
        if self.domain == "Negative":
            c_init_p1 = self.param.c_n_init(0, "phase 1")
            c_init_p2 = self.param.c_n_init(0, "phase 2")

        elif self.domain == "Positive":
            c_init_p1 = self.param.c_p_init(1, "phase 1")
            c_init_p2 = self.param.c_p_init(1, "phase 2")

        self.initial_conditions = {c_s_xav_p1: c_init_p1, c_s_xav_p2: c_init_p2}
