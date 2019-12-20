#
# Class for x-lumped thermal submodels with 2D current collectors
#
import pybamm

from .base_x_lumped import BaseModel


class CurrentCollector2D(BaseModel):
    """Class for x-lumped thermal submodel with 2D current collectors"""

    def __init__(self, param):
        super().__init__(param)

    def set_rhs(self, variables):
        T_av = variables["X-averaged cell temperature"]
        Q_av = variables["X-averaged total heating"]

        cooling_coeff = self._surface_cooling_coefficient()

        # Add boundary source term which accounts for surface cooling around
        # the edge of the domain in the weak formulation.
        # TODO: update to allow different cooling conditions at the tabs
        self.rhs = {
            T_av: (
                pybamm.laplacian(T_av)
                + self.param.B * pybamm.source(Q_av, T_av)
                + cooling_coeff * pybamm.source(T_av, T_av)
                - (self.param.h / self.param.delta)
                * pybamm.source(T_av, T_av, boundary=True)
            )
            / self.param.C_th
        }

    def set_boundary_conditions(self, variables):
        T_av = variables["X-averaged cell temperature"]
        # Dummy no flux boundary conditions since cooling at the the tabs is
        # accounted for in the boundary source term in the weak form of the
        # governing equation
        # TODO: update to allow different cooling conditions at the tabs
        self.boundary_conditions = {
            T_av: {
                "negative tab": (pybamm.Scalar(0), "Neumann"),
                "positive tab": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 2D current collector"""
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]

        Q_s_cn = self.param.sigma_cn_prime * pybamm.grad_squared(phi_s_cn)
        Q_s_cp = self.param.sigma_cp_prime * pybamm.grad_squared(phi_s_cp)
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """Computes the y-z average by integration over y and z"""
        return pybamm.yz_average(var)
