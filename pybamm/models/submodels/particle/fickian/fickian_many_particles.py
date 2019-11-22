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

        if self.domain == "Negative":
            x = pybamm.standard_spatial_vars.x_n
            R = pybamm.FunctionParameter( 'Negative particle distribution', x)
            variables.update({"Negative particle distribution":R})

        elif self.domain == "Positive":
            x = pybamm.standard_spatial_vars.x_p
            R = pybamm.FunctionParameter( 'Positive particle distribution', x)
            variables.update({"Positive particle distribution": R})
        return variables


    def set_rhs(self, variables):

        c, N, _ = self._unpack(variables)

        if self.domain == "Negative":
            R = pybamm.PrimaryBroadcast(variables['Negative particle distribution'], 'negative particle',)
            self.rhs = {c: -(1 / (R**2 * self.param.C_n)) * pybamm.div(N)}

        elif self.domain == "Positive":
            R = pybamm.PrimaryBroadcast(variables['Positive particle distribution'], 'positive particle',)
            self.rhs = {c: -(1 / (R**2 * self.param.C_p)) * pybamm.div(N)}


    def _unpack(self, variables):
        c_s = variables[self.domain + " particle concentration"]
        N_s = variables[self.domain + " particle flux"]
        j = variables[self.domain + " electrode interfacial current density"]

        return c_s, N_s, j
