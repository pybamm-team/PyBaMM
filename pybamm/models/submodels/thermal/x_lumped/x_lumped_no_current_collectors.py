#
# Class for lumped thermal submodel with no current collectors
#
import pybamm

from .base_x_lumped import BaseModel


class NoCurrentCollector(BaseModel):
    """
    Class for x-lumped thermal submodel without current collectors. Note: since
    there are no current collectors in this model, the electrochemical model
    must be 1D (x-direction only).

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def set_rhs(self, variables):
        T_av = variables["X-averaged cell temperature"]
        Q_av = variables["X-averaged total heating"]

        self.rhs = {
            T_av: (
                self.param.B * Q_av
                - (2 * self.param.h / (self.param.delta ** 2) / self.param.l) * T_av
            )
            / self.param.C_th
        }

    def _current_collector_heating(self, variables):
        """Returns zeros for current collector heat source terms"""
        Q_s_cn = pybamm.Scalar(0)
        Q_s_cp = pybamm.Scalar(0)
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """In 1D volume-averaged quantities are unchanged"""
        return var

    def _x_average(self, var, var_cn, var_cp):
        """
        Computes the x-average over the whole cell *not* including current
        collectors. This overwrites the default behaviour of 'base_thermal'.
        """
        return pybamm.x_average(var)
