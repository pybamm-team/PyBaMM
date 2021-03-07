#
# Class for many particles with Fickian diffusion
#
import pybamm
from .base_particle import BaseParticle


class FickianManyParticles(BaseParticle):
    """
    Class for molar conservation in many particles which employs Fick's law.

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

        variables = self._get_standard_concentration_variables(c_s)

        return variables

    def get_coupled_variables(self, variables):
        c_s = variables[self.domain + " particle concentration"]
        T = pybamm.PrimaryBroadcast(
            variables[self.domain + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )

        if self.domain == "Negative":
            N_s = -self.param.D_n(c_s, T) * pybamm.grad(c_s)

        elif self.domain == "Positive":
            N_s = -self.param.D_p(c_s, T) * pybamm.grad(c_s)

        variables.update(self._get_standard_flux_variables(N_s, N_s))
        variables.update(self._get_total_concentration_variables(variables))

        return variables

    def set_rhs(self, variables):
        c_s = variables[self.domain + " particle concentration"]
        N_s = variables[self.domain + " particle flux"]
        R = variables[self.domain + " particle radius"]

        if self.domain == "Negative":
            self.rhs = {c_s: -(1 / (R ** 2 * self.param.C_n)) * pybamm.div(N_s)}

        elif self.domain == "Positive":
            self.rhs = {c_s: -(1 / (R ** 2 * self.param.C_p)) * pybamm.div(N_s)}

    def set_boundary_conditions(self, variables):

        c_s = variables[self.domain + " particle concentration"]
        T = pybamm.PrimaryBroadcast(
            variables[self.domain + " electrode temperature"],
            c_s.domain,
        )
        j = variables[self.domain + " electrode interfacial current density"]
        R = variables[self.domain + " particle radius"]

        if self.domain == "Negative":
            rbc = (
                -self.param.C_n
                * j
                * R
                / self.param.a_R_n
                / pybamm.surf(self.param.D_n(c_s, T))
            )

        elif self.domain == "Positive":
            rbc = (
                -self.param.C_p
                * j
                * R
                / self.param.a_R_p
                / self.param.gamma_p
                / pybamm.surf(self.param.D_p(c_s, T))
            )

        self.boundary_conditions = {
            c_s: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }

    def set_initial_conditions(self, variables):
        c_s = variables[self.domain + " particle concentration"]

        if self.domain == "Negative":
            x_n = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_n, "negative particle"
            )
            c_init = self.param.c_n_init(x_n)

        elif self.domain == "Positive":
            x_p = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_p, "positive particle"
            )
            c_init = self.param.c_p_init(x_p)

        self.initial_conditions = {c_s: c_init}
