#
# Class for xyz-lumped thermal submodel with 2D current collectors
#
import pybamm

from .base_xyz_lumped import BaseModel


class CurrentCollector2D(BaseModel):
    """Class for xyz-lumped thermal submodel with 2D current collectors

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 2D current collector"""
        # For now we use the approximation that current collector heating
        # is uniform (see issue #765)
        i_boundary_cc = variables["Current collector current density"]
        Q_s_cn = i_boundary_cc / self.param.sigma_cn
        Q_s_cp = i_boundary_cc / self.param.sigma_cp
        return Q_s_cn, Q_s_cp

    def _surface_cooling_coefficient(self):
        """Returns the surface cooling coefficient in 2+1D"""
        # Note: assumes pouch cell geometry
        return (
            -2 * self.param.h / (self.param.delta ** 2) / self.param.l
            - 2 * (self.param.l_y + self.param.l_z) * self.param.h / self.param.delta
        )

    def _yz_average(self, var):
        """Computes the y-z average by integration over y and z"""
        return pybamm.yz_average(var)
