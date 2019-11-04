#
# Class for x-lumped thermal submodel with 0D current collector
#
from .base_x_lumped import BaseModel


class CurrentCollector0D(BaseModel):
    """Class for x-lumped thermal model with 0D current collectors"""

    def __init__(self, param):
        super().__init__(param)

    def set_rhs(self, variables):
        T_av = variables["X-averaged cell temperature"]
        Q_av = variables["X-averaged total heating"]

        cooling_coeff = self._surface_cooling_coefficient()

        self.rhs = {
            T_av: (self.param.B * Q_av + cooling_coeff * T_av) / self.param.C_th
        }

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 0D current collector"""
        i_boundary_cc = variables["Current collector current density"]
        Q_s_cn = i_boundary_cc ** 2 / self.param.sigma_cn
        Q_s_cp = i_boundary_cc ** 2 / self.param.sigma_cp
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """In 1D volume-averaged quantities are unchanged"""
        return var
