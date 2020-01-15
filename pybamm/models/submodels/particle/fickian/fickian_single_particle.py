#
# Class for a single particle with Fickian diffusion
#
import pybamm

from .base_fickian_particle import BaseModel


class SingleParticle(BaseModel):
    """Base class for molar conservation in a single x-averaged particle which employs
    Fick's law.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.particle.fickian.BaseModel`
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

        variables = self._get_standard_concentration_variables(c_s, c_s_xav)

        return variables

    def get_coupled_variables(self, variables):
        c_s_xav = variables[
            "X-averaged " + self.domain.lower() + " particle concentration"
        ]
        T_k_xav = pybamm.PrimaryBroadcast(
            variables["X-averaged " + self.domain.lower() + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )
        N_s_xav = self._flux_law(c_s_xav, T_k_xav)
        N_s = pybamm.SecondaryBroadcast(N_s_xav, [self._domain.lower() + " electrode"])

        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))

        return variables

    def set_rhs(self, variables):

        c, N, _ = self._unpack(variables)

        if self.domain == "Negative":
            self.rhs = {c: -(1 / self.param.C_n) * pybamm.div(N)}

        elif self.domain == "Positive":
            self.rhs = {c: -(1 / self.param.C_p) * pybamm.div(N)}

    def _unpack(self, variables):
        c_s_xav = variables[
            "X-averaged " + self.domain.lower() + " particle concentration"
        ]
        N_s_xav = variables["X-averaged " + self.domain.lower() + " particle flux"]
        j_av = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode interfacial current density"
        ]

        return c_s_xav, N_s_xav, j_av

    def set_initial_conditions(self, variables):
        """
        For single particle models, initial conditions can't depend on x so we
        arbitrarily evaluate them at x=0 in the negative electrode and x=1 in the
        positive electrode (they will usually be constant)
        """
        c, _, _ = self._unpack(variables)

        if self.domain == "Negative":
            c_init = self.param.c_n_init(0)

        elif self.domain == "Positive":
            c_init = self.param.c_p_init(1)

        self.initial_conditions = {c: c_init}
