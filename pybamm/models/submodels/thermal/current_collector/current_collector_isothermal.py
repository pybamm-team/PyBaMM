#
# Class for isothermal case which accounts for current collectors
#
import pybamm

from .base_current_collector_thermal import BaseModel


class Isothermal2D(BaseModel):
    """Class for isothermal submodel which accounts for current collectors.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.current_collector.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_av = pybamm.PrimaryBroadcast(0, "current collector")

        variables = self._get_standard_fundamental_variables(T_av)
        return variables

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 1D current collector"""
        # TODO: implement grad to calculate actual heating instead of average
        # approximate heating
        i_boundary_cc = variables["Current collector current density"]
        Q_s_cn = i_boundary_cc ** 2 / self.param.sigma_cn
        Q_s_cp = i_boundary_cc ** 2 / self.param.sigma_cp
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over z (no y-direction)"""
        return pybamm.z_average(var)

    def set_initial_conditions(self, variables):
        return {}


class Isothermal3D(Isothermal2D):
    """Class for 3D isothermal submodel"""

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
