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
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]
        Q_s_cn = self.param.sigma_cn_prime * pybamm.inner(
            pybamm.grad(phi_s_cn), pybamm.grad(phi_s_cn)
        )
        Q_s_cp = self.param.sigma_cp_prime * pybamm.inner(
            pybamm.grad(phi_s_cp), pybamm.grad(phi_s_cp)
        )
        return Q_s_cn, Q_s_cp

    def _surface_cooling_coefficient(self):
        """Returns the surface cooling coefficient in 1+1D"""
        # Note: assumes pouch cell geometry
        return (
            -2 * self.param.h / (self.param.delta ** 2) / self.param.l
            - self.param.l_z * self.param.h / self.param.delta
        )

    def _yz_average(self, var):
        """Computes the y-z average by integration over z (no y-direction)"""
        return pybamm.z_average(var)
