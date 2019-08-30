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

        # Add boundary source term which accounts for surface cooling around
        # the edge of the domain in  the weak formulation.
        # TODO: update to allow different cooling conditions at the tabs
        self.rhs = {
            T_av: (
                pybamm.laplacian(T_av)
                + self.param.B * Q_av
                - 2 * self.param.h / (self.param.delta ** 2) * T_av
                + self.param.h * pybamm.source(T_av, T_av, boundary=True)
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
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 2D current collector"""
        # TODO: implement grad to calculate actual heating instead of average
        # approximate heating
        i_boundary_cc = variables["Current collector current density"]
        Q_s_cn = i_boundary_cc ** 2 / self.param.sigma_cn
        Q_s_cp = i_boundary_cc ** 2 / self.param.sigma_cp
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over y and z"""
        return pybamm.yz_average(var)
