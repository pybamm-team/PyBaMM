#
# Class for lumped thermal submodel
#
import pybamm

from .base_thermal import BaseModel


class Lumped(BaseModel):
    """Class for lumped thermal submodel

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_av = pybamm.standard_variables.T_av
        T_n = pybamm.PrimaryBroadcast(T_av, ["negative electrode"])
        T_s = pybamm.PrimaryBroadcast(T_av, ["separator"])
        T_p = pybamm.PrimaryBroadcast(T_av, ["positive electrode"])
        T = pybamm.Concatenation(T_n, T_s, T_p)

        variables = self._get_standard_fundamental_variables(T)
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def _flux_law(self, T):
        """Fast x-direction heat diffusion (i.e. reached steady state)"""
        q = pybamm.FullBroadcast(
            pybamm.Scalar(0),
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )
        return q

    def _unpack(self, variables):
        T_av = variables["Average cell temperature"]
        q = variables["Heat flux"]
        Q_av = variables["Average total heating"]
        return T_av, q, Q_av

    def set_rhs(self, variables):
        T_av, _, Q_av = self._unpack(variables)

        self.rhs = {
            T_av: (
                self.param.B * Q_av - 2 * self.param.h / (self.param.delta ** 2) * T_av
            )
            / (self.param.C_th * self.param.rho)
        }
