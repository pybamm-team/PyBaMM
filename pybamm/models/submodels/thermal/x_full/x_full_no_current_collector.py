#
# Class for full thermal submodel
#
import pybamm

from .base_x_full import BaseModel


class NoCurrentCollector(BaseModel):
    """Class for full x-direction thermal submodel without current collectors

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.x_full.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def set_rhs(self, variables):
        T = variables["Cell temperature"]
        q = variables["Heat flux"]
        Q = variables["Total heating"]

        self.rhs = {
            T: (-pybamm.div(q) / self.param.delta ** 2 + self.param.B * Q)
            / (self.param.C_th * self.param.rho_k)
        }

    def set_boundary_conditions(self, variables):
        T = variables["Cell temperature"]
        T_n_left = pybamm.boundary_value(T, "left")
        T_p_right = pybamm.boundary_value(T, "right")

        self.boundary_conditions = {
            T: {
                "left": (self.param.h * T_n_left / self.param.lambda_n, "Neumann"),
                "right": (-self.param.h * T_p_right / self.param.lambda_p, "Neumann"),
            }
        }

    def _current_collector_heating(self, variables):
        """Returns zeros for current collector heat source terms"""
        Q_s_cn = pybamm.Scalar(0)
        Q_s_cp = pybamm.Scalar(0)
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """
        Computes the y-z average by integration over y and z
        In this case this is just equal to the input variable
        """
        return var

    def _x_average(self, var, var_cn, var_cp):
        """
        Computes the X-average over the whole cell *not* including current
        collectors. This overwrites the default behaviour of 'base_thermal'.
        """
        return pybamm.x_average(var)
