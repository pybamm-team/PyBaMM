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

    *Extends:* :class:`pybamm.BaseThermal`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_av = pybamm.Scalar(0)

        T_n = pybamm.Broadcast(T_av, ["negative electrode"])
        T_s = pybamm.Broadcast(T_av, ["separator"])
        T_p = pybamm.Broadcast(T_av, ["positive electrode"])

        T = pybamm.Concatenation(T_n, T_s, T_p)

        variables = self._get_standard_fundamental_variables(T, T_av)
        return variables

    def _flux_law(T):
        """Zero heat flux since temperature is constant"""
        q = pybamm.Broadcast(
            pybamm.Scalar(0), ["negative electrode", "separator", "positive electrode"]
        )
        return q

    def set_initial_conditions(self, variables):
        return {}

