#
# Class for thermal submodel in which the temperature is set externally
#
import pybamm

from .base_x_lumped import BaseModel


class SetTemperature1D(BaseModel):
    """Class for x-lumped thermal submodel which *doesn't* update the temperature.
    Instead, the temperature can be set (as a function of space) externally.
    Note, this model computes the heat generation terms for inspection after solve.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def set_rhs(self, variables):
        T_av = variables["X-averaged cell temperature"]

        # Dummy equation so that PyBaMM doesn't change the temperature during solve
        # i.e. d_T/d_t = 0. The (local) temperature is set externally between steps.
        self.rhs = {T_av: pybamm.Scalar(0)}

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
        """Computes the y-z average by integration over z (no y-direction)"""
        return pybamm.z_average(var)
