#
# Class for N+1D thermal submodel which accounts for current collectors
#
import pybamm

from .base_current_collector_thermal import BaseModel


class BaseNplus1D(BaseModel):
    """Base class for N+1D thermal submodel which accounts for current collectors.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.current_collector.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_av = pybamm.standard_variables.T_av

        variables = self._get_standard_fundamental_variables(T_av)
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def _unpack(self, variables):
        T_av = variables["X-averaged cell temperature"]
        q = variables["Heat flux"]
        Q_av = variables["X-averaged total heating"]
        return T_av, q, Q_av

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 0D current collector"""
        i_boundary_cc = variables["Current collector current density"]
        Q_s_cn = i_boundary_cc ** 2 / self.param.sigma_cn
        Q_s_cp = i_boundary_cc ** 2 / self.param.sigma_cp
        return Q_s_cn, Q_s_cp

    def set_rhs(self, variables):
        T_av, _, Q_av = self._unpack(variables)
        self.rhs = {
            T_av: (
                pybamm.laplacian(T_av)
                + self.param.B * Q_av
                - 2 * self.param.h / (self.param.delta ** 2) * T_av
            )
            / self.param.C_th
        }

    def set_boundary_conditions(self, variables):
        raise NotImplementedError


class Full1plus1D(BaseNplus1D):
    """Class for 1+1D thermal submodel"""

    def __init__(self, param):
        super().__init__(param)

    def set_boundary_conditions(self, variables):
        T_av, _, _ = self._unpack(variables)
        T_av_left = pybamm.boundary_value(T_av, "left")
        T_av_right = pybamm.boundary_value(T_av, "right")

        self.boundary_conditions = {
            T_av: {
                "left": (
                    -self.param.h * T_av_left / self.param.lambda_k / self.param.delta,
                    "Neumann",
                ),
                "right": (
                    -self.param.h * T_av_right / self.param.lambda_k / self.param.delta,
                    "Neumann",
                ),
            }
        }

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over z (no y-direction)"""
        return pybamm.z_average(var)


class Full2plus1D(BaseNplus1D):
    """Class for 1+1D thermal submodel"""

    def __init__(self, param):
        super().__init__(param)

    def set_boundary_conditions(self, variables):
        raise NotImplementedError

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over y and z"""
        return pybamm.yz_average(var)
