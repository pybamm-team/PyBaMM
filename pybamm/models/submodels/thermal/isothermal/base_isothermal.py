#
# Class for base isothermal submodel
#

import pybamm

from ..base_thermal import BaseThermal


class BaseIsothermal(BaseThermal):
    """Class for base isothermal submodel.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseThermal`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_x_av = pybamm.PrimaryBroadcast(0, "current collector")
        T_n = pybamm.PrimaryBroadcast(T_x_av, "negative electrode")
        T_s = pybamm.PrimaryBroadcast(T_x_av, "separator")
        T_p = pybamm.PrimaryBroadcast(T_x_av, "positive electrode")
        T = pybamm.Concatenation(T_n, T_s, T_p)

        T_cn = T_x_av
        T_cp = T_x_av

        variables = self._get_standard_fundamental_variables(T, T_cn, T_cp)
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def _flux_law(self, T):
        """Zero heat flux since temperature is constant"""
        q = pybamm.FullBroadcast(
            pybamm.Scalar(0),
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )
        return q
