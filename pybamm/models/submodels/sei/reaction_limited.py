#
# Class for reaction limited SEI growth
#
import pybamm
from .base_sei import BaseModel


class ReactionLimited(BaseModel):
    """Base class for reaction limited SEI growth.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        L_inner = pybamm.standard_variables.L_inner
        L_outer = pybamm.standard_variables.L_outer

        variables = self.get_standard_thickness_variables(L_inner, L_outer)

        return variables

    def get_coupled_variables(self, variables):
        c_s = variables[self.domain + " particle concentration"]
        T_k = pybamm.PrimaryBroadcast(
            variables[self.domain + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )

        if self.domain == "Negative":
            N_s = -self.param.D_n(c_s, T_k) * pybamm.grad(c_s)
        elif self.domain == "Positive":
            N_s = -self.param.D_p(c_s, T_k) * pybamm.grad(c_s)

        variables.update(self._get_standard_flux_variables(N_s, N_s))

        if self.domain == "Negative":
            x = pybamm.standard_spatial_vars.x_n
            R = pybamm.FunctionParameter("Negative particle distribution in x", x)
            variables.update({"Negative particle distribution in x": R})

        elif self.domain == "Positive":
            x = pybamm.standard_spatial_vars.x_p
            R = pybamm.FunctionParameter("Positive particle distribution in x", x)
            variables.update({"Positive particle distribution in x": R})

        return variables
