#
# Class for lumped thermal submodel
#
import pybamm

from .base_thermal import BaseThermal


class Lumped(BaseThermal):
    """Class for lumped thermal submodel

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    cc_dimension: int, optional
        The dimension of the current collectors. Can be 0 (default), 1 or 2.


    **Extends:** :class:`pybamm.thermal.BaseThermal`
    """

    def __init__(self, param, cc_dimension=0):
        self.cc_dimension = cc_dimension
        super().__init__(param)

    def get_fundamental_variables(self):

        T_vol_av = pybamm.standard_variables.T_vol_av
        T_x_av = pybamm.PrimaryBroadcast(T_vol_av, ["current collector"])

        T_cn = T_x_av
        T_n = pybamm.PrimaryBroadcast(T_x_av, "negative electrode")
        T_s = pybamm.PrimaryBroadcast(T_x_av, "separator")
        T_p = pybamm.PrimaryBroadcast(T_x_av, "positive electrode")
        T_cp = T_x_av

        variables = self._get_standard_fundamental_variables(
            T_cn, T_n, T_s, T_p, T_cp, T_x_av, T_vol_av
        )

        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def set_rhs(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        Q_vol_av = variables["Volume-averaged total heating"]
        T_amb = variables["Ambient temperature"]

        # Account for surface area to volume ratio in cooling coefficient
        # Note: assumes pouch cell geometry. The factor 1/delta^2 comes from
        # the choice of non-dimensionalisation.
        # TODO: allow for arbitrary surface area to volume ratio in order to model
        A = self.param.l_y * self.param.l_z
        V = self.param.l * self.param.l_y * self.param.l_z
        cooling_coeff = -2 * self.param.h * A / V / (self.param.delta ** 2)

        self.rhs = {
            T_vol_av: (self.param.B * Q_vol_av + cooling_coeff * (T_vol_av - T_amb))
            / (self.param.C_th * self.param.rho)
        }

    def set_initial_conditions(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        self.initial_conditions = {T_vol_av: self.param.T_init}

    def _current_collector_heating(self, variables):
        "Compute Ohmic heating in current collectors"
        # TODO: implement grad in 0D to return a scalar zero
        # TODO: implement grad_squared in other spatial methods so that the if
        # statement can be removed
        # In the limit of infinitely large current collector conductivity (i.e.
        # 0D current collectors), the Ohmic heating in the current collectors is
        # zero
        if self.cc_dimension == 0:
            Q_s_cn = pybamm.Scalar(0)
            Q_s_cp = pybamm.Scalar(0)
        # Otherwise we compute the Ohmic heating for 1 or 2D current collectors
        elif self.cc_dimension in [1, 2]:
            phi_s_cn = variables["Negative current collector potential"]
            phi_s_cp = variables["Positive current collector potential"]
            if self.cc_dimension == 1:
                Q_s_cn = self.param.sigma_cn_prime * pybamm.inner(
                    pybamm.grad(phi_s_cn), pybamm.grad(phi_s_cn)
                )
                Q_s_cp = self.param.sigma_cp_prime * pybamm.inner(
                    pybamm.grad(phi_s_cp), pybamm.grad(phi_s_cp)
                )
            elif self.cc_dimension == 2:
                # Inner not implemented in 2D -- have to call grad_squared directly
                Q_s_cn = self.param.sigma_cn_prime * pybamm.grad_squared(phi_s_cn)
                Q_s_cp = self.param.sigma_cp_prime * pybamm.grad_squared(phi_s_cp)
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """
        Computes the y-z average by integration over y and z
        In the 1D case this is just equal to the input variable
        """
        # TODO: change the behaviour of z_average and yz_average so the if statement
        # can be removed
        if self.cc_dimension == 0:
            return var
        elif self.cc_dimension == 1:
            return pybamm.z_average(var)
        elif self.cc_dimension == 2:
            return pybamm.yz_average(var)
