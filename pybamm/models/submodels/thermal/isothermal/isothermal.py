#
# Class for isothermal submodel
#
import pybamm

from ..base_thermal import BaseThermal


class Isothermal(BaseThermal):
    """Class for isothermal submodel.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseThermal`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_x_av = pybamm.PrimaryBroadcast(self.param.T_init, "current collector")
        T_n = pybamm.PrimaryBroadcast(T_x_av, "negative electrode")
        T_s = pybamm.PrimaryBroadcast(T_x_av, "separator")
        T_p = pybamm.PrimaryBroadcast(T_x_av, "positive electrode")
        T = pybamm.Concatenation(T_n, T_s, T_p)

        T_cn = T_x_av
        T_cp = T_x_av

        variables = self._get_standard_fundamental_variables(T, T_cn, T_cp)
        return variables

    def get_coupled_variables(self, variables):
        variables.update(
            {
                "Ohmic heating": pybamm.Scalar(0),
                "Ohmic heating [W.m-3]": pybamm.Scalar(0),
                "Irreversible electrochemical heating": pybamm.Scalar(0),
                "Irreversible electrochemical heating [W.m-3]": pybamm.Scalar(0),
                "Reversible heating": pybamm.Scalar(0),
                "Reversible heating [W.m-3]": pybamm.Scalar(0),
                "Total heating": pybamm.Scalar(0),
                "Total heating [W.m-3]": pybamm.Scalar(0),
                "X-averaged total heating": pybamm.Scalar(0),
                "X-averaged total heating [W.m-3]": pybamm.Scalar(0),
                "Volume-averaged total heating": pybamm.Scalar(0),
                "Volume-averaged total heating [W.m-3]": pybamm.Scalar(0),
            }
        )
        return variables

    def _flux_law(self, T):
        """Zero heat flux since temperature is constant"""
        q = pybamm.FullBroadcast(
            pybamm.Scalar(0),
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )
        return q

    def _yz_average(self, var):
        """
        Temperature is uniform and heat source terms are zero, so the average
        returns the input variable.
        """
        return var

    def _x_average(self, var, var_cn, var_cp):
        """
        Temperature is uniform and heat source terms are zero, so the average
        returns zeros broadcasted onto the current collector domain.
        """
        return pybamm.PrimaryBroadcast(0, "current collector")
