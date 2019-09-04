#
# Class for x-lumped thermal submodel with 1D current collector
#
import pybamm

from .base_x_lumped import BaseModel


class CurrentCollector1D(BaseModel):
    """Class for x-lumped thermal model with 1D current collectors"""

    def __init__(self, param):
        super().__init__(param)

    def set_rhs(self, variables):
        T_av = variables["X-averaged cell temperature"]
        Q_av = variables["X-averaged total heating"]

        self.rhs = {
            T_av: (
                pybamm.laplacian(T_av)
                + self.param.B * Q_av
                - 2 * self.param.h / (self.param.delta ** 2) * T_av
            )
            / self.param.C_th
        }

    def set_boundary_conditions(self, variables):
        T_av = variables["X-averaged cell temperature"]
        T_av_left = pybamm.boundary_value(T_av, "left")
        T_av_right = pybamm.boundary_value(T_av, "right")

        self.boundary_conditions = {
            T_av: {
                "left": (
                    self.param.h * T_av_left / self.param.lambda_k / self.param.delta,
                    "Neumann",
                ),
                "right": (
                    -self.param.h * T_av_right / self.param.lambda_k / self.param.delta,
                    "Neumann",
                ),
            }
        }

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 1D current collector"""
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]
        Q_s_cn = self.param.sigma_cn_prime * pybamm.inner(
            pybamm.grad(phi_s_cn), pybamm.grad(phi_s_cn)
        )
        Q_s_cp = self.param.sigma_cp_prime * pybamm.inner(
            pybamm.grad(phi_s_cp), pybamm.grad(phi_s_cp)
        )
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over z (no y-direction)"""
        return pybamm.z_average(var)
