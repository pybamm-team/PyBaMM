#
# Class for many particles with Fickian diffusion
#
import pybamm
from .base_fickian_particle import BaseModel


class ManyParticles(BaseModel):
    """Base class for molar conservation in many particles which employs
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
            c_s = pybamm.standard_variables.c_s_n

        elif self.domain == "Positive":
            c_s = pybamm.standard_variables.c_s_p

        # TODO: implement c_s_xav for Fickian many particles (tricky because this
        # requires averaging a secondary domain)
        variables = self._get_standard_concentration_variables(c_s, c_s)

        return variables

    def get_coupled_variables(self, variables):
        c_s = variables[self.domain + " particle concentration"]
        T_k = pybamm.PrimaryBroadcast(
            variables[self.domain + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )

        N_s = self._flux_law(c_s, T_k)

        variables.update(self._get_standard_flux_variables(N_s, N_s))
        return variables

    def set_boundary_conditions(self, variables):

        c_s = variables[self.domain + " particle concentration"]
        c_s_surf = variables[self.domain + " particle surface concentration"]
        T_k = variables[self.domain + " electrode temperature"]
        j = variables[self.domain + " electrode interfacial current density"]

        if self.domain == "Negative":
            rbc = -self.param.C_n * j / self.param.a_n / self.param.D_n(c_s_surf, T_k)

        elif self.domain == "Positive":
            rbc = (
                -self.param.C_p
                * j
                / self.param.a_p
                / self.param.gamma_p
                / self.param.D_p(c_s_surf, T_k)
            )

        self.boundary_conditions = {
            c_s: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }

    def _unpack(self, variables):
        c_s = variables[self.domain + " particle concentration"]
        N_s = variables[self.domain + " particle flux"]
        j = variables[self.domain + " electrode interfacial current density"]

        return c_s, N_s, j
