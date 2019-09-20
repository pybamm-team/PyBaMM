#
# Class for lumped thermal submodel
#
import pybamm
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
        # Note: need to get the total heating and avergae over the negative
        # electrode,separator and positive electrode. The variable ["X-averaged
        # total heating"] is the avergae in x *including* the current collectors
        # so results in an underprediction of heating when compared with the submodel
        # "x_full_no_current_collector". For the same reason, we use
        # `pybamm.x_average(T)` in the cooling term. The equation is still for the
        # x-averaged temperature `T_av`, which gets broadcasted over the entire cell.
        T = variables["Cell temperature"]
        T_av = variables["X-averaged cell temperature"]
        Q = variables["Total heating"]
        Q_av = pybamm.x_average(Q)

        self.rhs = {
            T_av: (
                self.param.B * Q_av
                - (2 * self.param.h / (self.param.delta ** 2) / self.param.l)
                * pybamm.x_average(T)
            )
            / self.param.C_th
        }

    def _current_collector_heating(self, variables):
        """Returns zeros for current collector heat source terms"""
        Q_s_cn = pybamm.Scalar(0)
        Q_s_cp = pybamm.Scalar(0)
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """Computes the y-z avergage (just the variable when no current collectors)"""
        return var
