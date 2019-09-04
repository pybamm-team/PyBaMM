#
# Class for x-lumped thermal submodel with 0D current collector
#
from .base_x_lumped import BaseModel


class CurrentCollector0D(BaseModel):
    """Class for x-lumped thermal model with 0D current collectors"""

    def __init__(self, param):
        super().__init__(param)

    def set_rhs(self, variables):
        T_volume_av = variables["Volume-averaged cell temperature"]
        Q_volume_av = variables["Volume-averaged total heating"]

        self.rhs = {
            T_volume_av: (
                self.param.B * Q_volume_av
                + -2 * self.param.h / (self.param.delta ** 2) * T_volume_av
            )
            / self.param.C_th
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
