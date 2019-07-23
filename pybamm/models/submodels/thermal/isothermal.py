#
# Class for isothermal case
#
import pybamm

from .base_thermal import BaseModel


class Isothermal(BaseModel):
    """Class for isothermal submodel

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_n = pybamm.FullBroadcast(0, "negative electrode", "current collector")
        T_s = pybamm.FullBroadcast(0, "separator", "current collector")
        T_p = pybamm.FullBroadcast(0, "positive electrode", "current collector")
        T = pybamm.Concatenation(T_n, T_s, T_p)

        variables = self._get_standard_fundamental_variables(T)
        return variables

    def _flux_law(self, T):
        """Zero heat flux since temperature is constant"""
        q = pybamm.Broadcast(
            pybamm.Scalar(0), ["negative electrode", "separator", "positive electrode"]
        )
        return q

    def set_initial_conditions(self, variables):
        return {}
