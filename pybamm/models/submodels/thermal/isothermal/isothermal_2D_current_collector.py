#
# Class for isothermal case which accounts for current collectors
#
import pybamm

from .base_isothermal import BaseModel


class CurrentCollector2D(BaseModel):
    """Class for isothermal submodel with a 2D current collector"""

    def __init__(self, param):
        super().__init__(param)

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
