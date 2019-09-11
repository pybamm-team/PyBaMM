#
# Base class for xyz-lumped thermal submodels
#
import pybamm

from ..base_thermal import BaseThermal


class BaseModel(BaseThermal):
    """Base class for xyz-lumped thermal submodel

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

        T_n = pybamm.PrimaryBroadcast(T_x_av, "negative electrode")
        T_s = pybamm.PrimaryBroadcast(T_x_av, "separator")
        T_p = pybamm.PrimaryBroadcast(T_x_av, "positive electrode")
        T = pybamm.Concatenation(T_n, T_s, T_p)

        T_cn = T_x_av
        T_cp = T_x_av

        variables = self._get_standard_fundamental_variables(T, T_cn, T_cp)

        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def set_rhs(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        Q_vol_av = variables["Volume-averaged total heating"]

        cooling_coeff = self._surface_cooling_coefficient()

        self.rhs = {
            T_vol_av: (self.param.B * Q_vol_av + cooling_coeff * T_vol_av)
            / self.param.C_th
        }

    def _flux_law(self, T):
        """Fast heat diffusion (temperature has no spatial dependence)"""
        q = pybamm.FullBroadcast(
            pybamm.Scalar(0),
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )
        return q

    def set_initial_conditions(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        self.initial_conditions = {T_vol_av: self.param.T_init}
