#
# Class for a single particle with Fickian diffusion
#
import pybamm

from .base_particle import BaseParticle


class FickianSingleParticle(BaseParticle):
    """
    Class for molar conservation in a single x-averaged particle which employs
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

        variables = self._get_standard_concentration_variables(c_s, c_s_xav=c_s_xav)

        return variables

    def get_coupled_variables(self, variables):
        c_s_xav = variables[
            "X-averaged " + self.domain.lower() + " particle concentration"
        ]
        T_xav = pybamm.PrimaryBroadcast(
            variables["X-averaged " + self.domain.lower() + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )

        if self.domain == "Negative":
            N_s_xav = -self.param.D_n(c_s_xav, T_xav) * pybamm.grad(c_s_xav)

        elif self.domain == "Positive":
            N_s_xav = -self.param.D_p(c_s_xav, T_xav) * pybamm.grad(c_s_xav)

        N_s = pybamm.SecondaryBroadcast(N_s_xav, [self._domain.lower() + " electrode"])

        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))
        variables.update(self._get_total_concentration_variables(variables))

        return variables

    def set_rhs(self, variables):
        c_s_xav = variables[
            "X-averaged " + self.domain.lower() + " particle concentration"
        ]
        N_s_xav = variables["X-averaged " + self.domain.lower() + " particle flux"]

        if self.domain == "Negative":
            self.rhs = {c_s_xav: -(1 / self.param.C_n) * pybamm.div(N_s_xav)}

        elif self.domain == "Positive":
            self.rhs = {c_s_xav: -(1 / self.param.C_p) * pybamm.div(N_s_xav)}

    def set_boundary_conditions(self, variables):
        c_s_xav = variables[
            "X-averaged " + self.domain.lower() + " particle concentration"
        ]
        T_xav = pybamm.PrimaryBroadcast(
            variables["X-averaged " + self.domain.lower() + " electrode temperature"],
            c_s_xav.domain[0],
        )
        j_xav = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode interfacial current density"
        ]

        if self.domain == "Negative":
            rbc = (
                -self.param.C_n
                * j_xav
                / self.param.a_R_n
                / pybamm.surf(self.param.D_n(c_s_xav, T_xav))
            )

        elif self.domain == "Positive":
            rbc = (
                -self.param.C_p
                * j_xav
                / self.param.a_R_p
                / self.param.gamma_p
                / pybamm.surf(self.param.D_p(c_s_xav, T_xav))
            )

        self.boundary_conditions = {
            c_s_xav: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }

    def set_initial_conditions(self, variables):
        """
        For single particle models, initial conditions can't depend on x so we
        arbitrarily set the initial values of the single particles to be given
        by the values at x=0 in the negative electrode and x=1 in the
        positive electrode. Typically, supplied initial conditions are uniform
        x.
        """
        c_s_xav = variables[
            "X-averaged " + self.domain.lower() + " particle concentration"
        ]

        if self.domain == "Negative":
            c_init = self.param.c_n_init(0)

        elif self.domain == "Positive":
            c_init = self.param.c_p_init(1)

        self.initial_conditions = {c_s_xav: c_init}
