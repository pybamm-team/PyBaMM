#
# Class for lumped thermal submodel
#
import pybamm

from .base_thermal import BaseThermal


class Lumped(BaseThermal):
    """Class for lumped thermal submodel for use with 1D electrochemical models

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseThermal`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_x_av = pybamm.standard_variables.T_av

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
        T_av = variables["X-averaged cell temperature"]
        Q_av = variables["X-averaged total heating"]
        T_amb = variables["Ambient temperature"]

        # Account for surface area to volume ratio in cooling coefficient
        # Note: assumes pouch cell geometry. The factor 1/delta^2 comes from
        # the choice of non-dimensionalisation.
        A = self.param.l_y * self.param.l_z
        V = self.param.l * self.param.l_y * self.param.l_z
        cooling_coeff = -2 * self.param.h * A / V / (self.param.delta ** 2)

        self.rhs = {
            T_av: (self.param.B * Q_av + cooling_coeff * (T_av - T_amb))
            / (self.param.C_th * self.param.rho)
        }

    def set_initial_conditions(self, variables):
        T = variables["X-averaged cell temperature"]
        self.initial_conditions = {T: self.param.T_init}

    def _current_collector_heating(self, variables):
        """
        In the limit of infinitely large current collector conductivity, the
        Ohmic heating in the current collectors in zero
        """
        Q_s_cn = pybamm.Scalar(0)
        Q_s_cp = pybamm.Scalar(0)
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """
        Computes the y-z average by integration over y and z
        In the 1D case this is just equal to the input variable
        """
        return var
