#
# Class for isothermal submodel
#
import pybamm

from .base_thermal import BaseThermal


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
        T_amb = self.param.T_amb(pybamm.t * self.param.timescale)
        T_x_av = pybamm.PrimaryBroadcast(T_amb, "current collector")
        T_vol_av = pybamm.PrimaryBroadcast(T_amb, "current collector")
        T_cn = T_x_av
        T_n = pybamm.PrimaryBroadcast(T_x_av, "negative electrode")
        T_s = pybamm.PrimaryBroadcast(T_x_av, "separator")
        T_p = pybamm.PrimaryBroadcast(T_x_av, "positive electrode")
        T_cp = T_x_av

        variables = self._get_standard_fundamental_variables(
            T_cn, T_n, T_s, T_p, T_cp, T_x_av, T_vol_av
        )

        return variables

    def get_coupled_variables(self, variables):
        ieh = "irreversible electrochemical heating"
        variables.update(
            {
                "Ohmic heating": pybamm.Scalar(0),
                "Ohmic heating [W.m-3]": pybamm.Scalar(0),
                "X-averaged Ohmic heating": pybamm.Scalar(0),
                "X-averaged Ohmic heating [W.m-3]": pybamm.Scalar(0),
                "Volume-averaged Ohmic heating": pybamm.Scalar(0),
                "Volume-averaged Ohmic heating [W.m-3]": pybamm.Scalar(0),
                "Irreversible electrochemical heating": pybamm.Scalar(0),
                "Irreversible electrochemical heating [W.m-3]": pybamm.Scalar(0),
                "X-averaged " + ieh: pybamm.Scalar(0),
                "X-averaged " + ieh + " [W.m-3]": pybamm.Scalar(0),
                "Volume-averaged " + ieh: pybamm.Scalar(0),
                "Volume-averaged " + ieh + "[W.m-3]": pybamm.Scalar(0),
                "Reversible heating": pybamm.Scalar(0),
                "Reversible heating [W.m-3]": pybamm.Scalar(0),
                "X-averaged reversible heating": pybamm.Scalar(0),
                "X-averaged reversible heating [W.m-3]": pybamm.Scalar(0),
                "Volume-averaged reversible heating": pybamm.Scalar(0),
                "Volume-averaged reversible heating [W.m-3]": pybamm.Scalar(0),
                "Total heating": pybamm.Scalar(0),
                "Total heating [W.m-3]": pybamm.Scalar(0),
                "X-averaged total heating": pybamm.Scalar(0),
                "X-averaged total heating [W.m-3]": pybamm.Scalar(0),
                "Volume-averaged total heating": pybamm.Scalar(0),
                "Volume-averaged total heating [W.m-3]": pybamm.Scalar(0),
            }
        )
        return variables
