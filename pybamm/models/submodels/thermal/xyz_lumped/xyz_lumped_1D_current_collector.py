#
# Class for xyz-lumped thermal submodel with 1D current collectors
#
import pybamm

from .base_xyz_lumped import BaseModel


class CurrentCollector1D(BaseModel):
    """Class for xyz-lumped thermal submodel with 1D current collectors

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 1D current collector"""
        # TODO: implement grad to calculate actual heating instead of average
        # approximate heating
        i_boundary_cc = variables["Current collector current density"]
        Q_s_cn = i_boundary_cc ** 2 / self.param.sigma_cn
        Q_s_cp = i_boundary_cc ** 2 / self.param.sigma_cp
        return Q_s_cn, Q_s_cp

    def _surface_cooling_coefficient(self):
        """Returns the surface cooling coefficient in 1+1D"""
        return (
            -2 * self.param.h / (self.param.delta ** 2)
            - 2 * self.param.l_z * self.param.h / self.param.delta
        )

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over z (no y-direction)"""
        return pybamm.z_average(var)
