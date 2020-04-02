#
# Classes for lumped thermal submodels for use in the "N+1D" pouch cell models
#
import pybamm

from ..base_thermal import BaseThermal


class LumpedPouchCell(BaseThermal):
    """Base class for lumped thermal submodels for use in the "N+1D" pouch cell
    models.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_vol_av = pybamm.standard_variables.T_vol_av
        T_x_av = pybamm.PrimaryBroadcast(T_vol_av, ["current collector"])

        T_cn = T_x_av
        T_n = pybamm.PrimaryBroadcast(T_x_av, "negative electrode")
        T_s = pybamm.PrimaryBroadcast(T_x_av, "separator")
        T_p = pybamm.PrimaryBroadcast(T_x_av, "positive electrode")
        T_cp = T_x_av

        variables = self._get_standard_fundamental_variables(T_cn, T_n, T_s, T_p, T_cp)

        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def set_rhs(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        Q_vol_av = variables["Volume-averaged total heating"]
        T_amb = variables["Ambient temperature"]

        cooling_coeff = self._surface_cooling_coefficient()

        self.rhs = {
            T_vol_av: (self.param.B * Q_vol_av + cooling_coeff * (T_vol_av - T_amb))
            / (self.param.C_th * self.param.rho)
        }

    def set_initial_conditions(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        self.initial_conditions = {T_vol_av: self.param.T_init}


class LumpedPouchCell1D(LumpedPouchCell):
    """Class for lumped thermal submodel for use in 1+1D pouch cell models

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    **Extends:** :class:`pybamm.thermal.pouch_cell.LumpedPouchCell`
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
        """Returns the surface cooling coefficient for the pouch cell in 1+1D"""
        return (
            -2 * self.param.h / (self.param.delta ** 2) / self.param.l
            - self.param.l_z * self.param.h / self.param.delta
        )

    def _yz_average(self, var):
        """Computes the y-z average by integration over z (no y-direction)"""
        return pybamm.z_average(var)


class LumpedPouchCell2D(LumpedPouchCell):
    """Class for lumped thermal submodel for use in 2+1D pouch cell models

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    **Extends:** :class:`pybamm.thermal.pouch_cell.LumpedPouchCell`
    """

    def __init__(self, param):
        super().__init__(param)

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 2D current collector"""
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]

        Q_s_cn = self.param.sigma_cn_prime * pybamm.grad_squared(phi_s_cn)
        Q_s_cp = self.param.sigma_cp_prime * pybamm.grad_squared(phi_s_cp)
        return Q_s_cn, Q_s_cp

    def _surface_cooling_coefficient(self):
        """Returns the surface cooling coefficient for the pouch cell in 2+1D"""
        return (
            -2 * self.param.h / (self.param.delta ** 2) / self.param.l
            - 2 * (self.param.l_y + self.param.l_z) * self.param.h / self.param.delta
        )

    def _yz_average(self, var):
        """Computes the y-z average by integration over y and z"""
        return pybamm.yz_average(var)
