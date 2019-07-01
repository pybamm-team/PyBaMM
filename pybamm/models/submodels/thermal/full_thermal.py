#
# Class for full thermal submodel
#
import pybamm

from .base_thermal import BaseModel


class Full(BaseModel):
    """Class for full thermal submodel

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
        T_av = pybamm.average(T)

        variables = self._get_standard_fundamental_variables(T, T_av)
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def _flux_law(self, T):
        """Fourier's law for heat transfer"""
        q = -self.param.lambda_k * pybamm.grad(T)
        return q

    def _unpack(self, variables):
        T = variables["Cell temperature"]
        q = variables["Heat flux"]
        Q = variables["Total heating"]
        return T, q, Q

    def set_rhs(self, variables):
        T, q, Q = self._unpack(variables)

        self.rhs = {
            T: (-pybamm.div(q) + self.param.delta ** 2 * self.param.B * Q)
            / (self.param.delta ** 2 * self.param.C_th * self.param.rho_k)
        }

    def set_boundary_conditions(self, variables):
        T, _, _ = self._unpack(variables)
        T_n_left = pybamm.boundary_value(T, "left")
        T_p_right = pybamm.boundary_value(T, "right")

        self.boundary_conditions = {
            T: {
                "left": (-self.param.h * T_n_left / self.param.lambda_k, "Neumann"),
                "right": (-self.param.h * T_p_right / self.param.lambda_k, "Neumann"),
            }
        }
