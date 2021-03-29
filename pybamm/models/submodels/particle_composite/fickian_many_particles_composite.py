#
# Class for many particles with Fickian diffusion
#
import pybamm
from .base_particle_composite import BaseParticleComposite


class FickianManyParticlesComposite(BaseParticleComposite):
    """
    Class for molar conservation in many composite particles which employs Fick's law.

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
            c_s = pybamm.standard_variables.c_s_n

        elif self.domain == "Positive":
            c_s = pybamm.standard_variables.c_s_p

        variables = self._get_standard_concentration_variables(
            c_s,
            phase="phase 1",
        )
        variables.update(
            self._get_standard_concentration_variables(c_s, phase="phase 2")
        )
        variables.update(self._get_standard_concentration_variables(c_s))
        return variables

    def get_coupled_variables(self, variables):
        if self.domain == "Negative":
            p1_name = " of phase 1"
            p2_name = " of phase 2"
        elif self.domain == "Positive":
            p1_name = " of phase 1"
            p2_name = " of phase 2"

        c_s_p1 = variables[f"{self.domain} particle concentration{p1_name}"]
        c_s_p2 = variables[f"{self.domain} particle concentration{p2_name}"]
        T = pybamm.PrimaryBroadcast(
            variables[self.domain + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )

        if self.domain == "Negative":
            N_s_p1 = -self.param.D_n(c_s_p1, T, "phase 1") * pybamm.grad(c_s_p1)
            N_s_p2 = -self.param.D_n(c_s_p2, T, "phase 2") * pybamm.grad(c_s_p2)
            N_s = N_s_p1 * self.param.V_n_p1 + N_s_p2 * self.param.V_n_p2
        elif self.domain == "Positive":
            N_s_p1 = -self.param.D_p(c_s_p1, T, "phase 1") * pybamm.grad(c_s_p1)
            N_s_p2 = -self.param.D_p(c_s_p2, T, "phase 2") * pybamm.grad(c_s_p2)
            N_s = N_s_p1 * self.param.V_p_p1 + N_s_p2 * self.param.V_p_p2

        variables.update(
            self._get_standard_flux_variables(N_s_p1, N_s_p1, phase="phase 1")
        )
        variables.update(
            self._get_standard_flux_variables(N_s_p2, N_s_p2, phase="phase 2")
        )
        variables.update(self._get_standard_flux_variables(N_s, N_s))
        variables.update(
            self._get_total_concentration_variables(variables, phase="phase 1")
        )
        variables.update(
            self._get_total_concentration_variables(variables, phase="phase 2")
        )
        return variables

    def set_rhs(self, variables):
        if self.domain == "Negative":
            p1_name = " of phase 1"
            p2_name = " of phase 2"
            C_s_p1 = self.param.C_n_p1
            C_s_p2 = self.param.C_n_p2
        elif self.domain == "Positive":
            p1_name = " of phase 1"
            p2_name = " of phase 2"
            C_s_p1 = self.param.C_p_p1
            C_s_p2 = self.param.C_p_p2

        c_s_p1 = variables[f"{self.domain} particle concentration{p1_name}"]
        c_s_p2 = variables[f"{self.domain} particle concentration{p2_name}"]
        N_s_p1 = variables[f"{self.domain} particle flux{p1_name}"]
        N_s_p2 = variables[f"{self.domain} particle flux{p2_name}"]
        R_p1 = variables[f"{self.domain} particle radius{p1_name}"]
        R_p2 = variables[f"{self.domain} particle radius{p2_name}"]
        self.rhs = {
            c_s_p1: -(1 / (R_p1 ** 2 * C_s_p1)) * pybamm.div(N_s_p1),
            c_s_p2: -(1 / (R_p2 ** 2 * C_s_p1)) * pybamm.div(N_s_p2),
        }

    def set_boundary_conditions(self, variables):
        if self.domain == "Negative":
            p1_name = " of phase 1"
            p2_name = " of phase 2"
        elif self.domain == "Positive":
            p1_name = " of phase 1"
            p2_name = " of phase 2"
        c_s_p1 = variables[f"{self.domain} particle concentration{p1_name}"]
        c_s_surf_p1 = variables[
            f"{self.domain} particle surface concentration{p1_name}"
        ]
        c_s_p2 = variables[f"{self.domain} particle concentration{p2_name}"]
        c_s_surf_p2 = variables[
            f"{self.domain} particle surface concentration{p2_name}"
        ]
        T = variables[self.domain + " electrode temperature"]
        j_p1 = variables[
            f"{self.domain} electrode interfacial current density{p1_name}"
        ]
        j_p2 = variables[
            f"{self.domain} electrode interfacial current density{p2_name}"
        ]
        R_p1 = variables[f"{self.domain} particle radius{p1_name}"]
        R_p2 = variables[f"{self.domain} particle radius{p2_name}"]

        if self.domain == "Negative":
            rbc_p1 = (
                -self.param.C_n_p1
                * j_p1
                * R_p1
                / self.param.a_R_n_p1
                / self.param.D_n(c_s_surf_p1, T, "phase 1")
            )
            rbc_p2 = (
                -self.param.C_n_p2
                * j_p2
                * R_p2
                / self.param.a_R_n_p2
                / self.param.D_n(c_s_surf_p2, T, "phase 2")
            )

        elif self.domain == "Positive":
            rbc_p1 = (
                -self.param.C_p_p1
                * j_p1
                * R_p1
                / self.param.a_R_p_p1
                / self.param.gamma_p
                / self.param.D_p(c_s_surf_p1, T, "phase 1")
            )
            rbc_p2 = (
                -self.param.C_p_p2
                * j_p2
                * R_p2
                / self.param.a_R_p_p2
                / self.param.gamma_p
                / self.param.D_p(c_s_surf_p2, T, "phase 2")
            )

        self.boundary_conditions = {
            c_s_p1: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (rbc_p1, "Neumann"),
            },
            c_s_p2: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (rbc_p2, "Neumann"),
            },
        }

    def set_initial_conditions(self, variables):
        if self.domain == "Negative":
            p1_name = " of phase 1"
            p2_name = " of phase 2"
        elif self.domain == "Positive":
            p1_name = " of phase 1"
            p2_name = " of phase 2"
        c_s_p1 = variables[f"{self.domain} particle concentration{p1_name}"]
        c_s_p2 = variables[f"{self.domain} particle concentration{p2_name}"]

        if self.domain == "Negative":
            x_n = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_n, "negative particle"
            )
            c_init_p1 = self.param.c_n_init(x_n, "phase 1")
            c_init_p2 = self.param.c_n_init(x_n, "phase 2")

        elif self.domain == "Positive":
            x_p = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_p, "positive particle"
            )
            c_init_p1 = self.param.c_p_init(x_p, "phase 1")
            c_init_p2 = self.param.c_p_init(x_p, "phase 2")

        self.initial_conditions = {c_s_p1: c_init_p1, c_s_p2: c_init_p2}
