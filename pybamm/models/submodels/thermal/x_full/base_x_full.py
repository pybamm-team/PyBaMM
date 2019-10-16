#
# Base class for x-full thermal submodels
#
import pybamm

from ..base_thermal import BaseThermal


class BaseModel(BaseThermal):
    """Base class for full x-direction thermal submodels.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        T = pybamm.standard_variables.T
        T_cn = pybamm.BoundaryValue(T, "left")
        T_cp = pybamm.BoundaryValue(T, "right")
        variables = self._get_standard_fundamental_variables(T, T_cn, T_cp)
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def _flux_law(self, T):
        """Fourier's law for heat transfer"""
        q = -self.param.lambda_k * pybamm.grad(T)
        return q

    def set_initial_conditions(self, variables):
        T = variables["Cell temperature"]
        self.initial_conditions = {T: self.param.T_init}
