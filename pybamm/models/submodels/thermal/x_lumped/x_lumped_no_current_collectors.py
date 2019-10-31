#
# Class for lumped thermal submodel with no current collectors
#
import pybamm

from .base_x_lumped import BaseModel


class NoCurrentCollector(BaseModel):
    """
    Class for x-lumped thermal submodel without current collectors. Note: since
    there are no current collectors in this model, the electrochemical model
    must be 1D (x-direction only).

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
        Q_av = variables["X-averaged total heating"]

        # Get effective properties
        rho_eff, _ = self._effective_properties()
        cooling_coeff = self._surface_cooling_coefficient()

        self.rhs = {
            T_av: (self.param.B * Q_av + cooling_coeff * T_av)
            / (self.param.C_th * rho_eff)
        }

    def _current_collector_heating(self, variables):
        """Returns zeros for current collector heat source terms"""
        Q_s_cn = pybamm.Scalar(0)
        Q_s_cp = pybamm.Scalar(0)
        return Q_s_cn, Q_s_cp

    def _surface_cooling_coefficient(self):
        """
        Returns the surface cooling coefficient in the absence of current
        collectors.
        """
        # Account for surface area to volume ratio in cooling coefficient
        # Note: assumes pouch cell geometry and volume doesn't include current
        # collectors
        A = self.param.l_y * self.param.l_z
        V = self.param.l_x * self.param.l_y * self.param.l_z
        return -2 * self.param.h * A / V / (self.param.delta ** 2)

    def _yz_average(self, var):
        """In 1D volume-averaged quantities are unchanged"""
        return var

    def _x_average(self, var, var_cn, var_cp):
        """
        Computes the X-average over the whole cell *not* including current
        collectors. This overwrites the default behaviour of 'base_thermal'.
        """
        return pybamm.x_average(var)

    def _effective_properties(self):
        """
        Computes the effective effective product of density and specific heat, and
        effective thermal conductivity, respectively. This overwrites the default
        behaviour of 'base_thermal'.
        """
        l_n = self.param.l_n
        l_s = self.param.l_s
        l_p = self.param.l_p
        rho_n = self.param.rho_n
        rho_s = self.param.rho_s
        rho_p = self.param.rho_p
        lambda_n = self.param.lambda_n
        lambda_s = self.param.lambda_s
        lambda_p = self.param.lambda_p

        rho_eff = (rho_n * l_n + rho_s * l_s + rho_p * l_p) / (l_n + l_n + l_p)
        lambda_eff = (lambda_n * l_n + lambda_s * l_s + lambda_p * l_p) / (
            l_n + l_n + l_p
        )

        return rho_eff, lambda_eff
