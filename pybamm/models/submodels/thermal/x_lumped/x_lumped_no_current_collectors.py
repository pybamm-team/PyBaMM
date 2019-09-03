#
# Class for lumped thermal submodel
#
from .base_x_lumped import BaseModel


class NoCurrentCollector(BaseModel):
    """Class for x-lumped thermal submodel without current collectors

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
                self.param.B * Q_av - 2 * self.param.h / (self.param.delta ** 2) * T_av
            )
            / (self.param.C_th * self.param.rho)
        }

    def _current_collector_heating(self, variables):
        """Returns zeros for current collector heat source terms"""
        Q_s_cn = 0
        Q_s_cp = 0
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """Computes the y-z avergage (just the variable when no current collectors)"""
        return var
