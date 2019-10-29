#
# Base class for x-lumped thermal submodels
#
import pybamm

from ..base_thermal import BaseThermal


class BaseModel(BaseThermal):
    """Base class for x-lumped thermal submodel

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_x_av = pybamm.standard_variables.T_av
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

    def _flux_law(self, T):
        """Fast heat diffusion (temperature has no spatial dependence)"""
        q = pybamm.FullBroadcast(
            pybamm.Scalar(0),
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )
        return q

    def set_initial_conditions(self, variables):
        T = variables["X-averaged cell temperature"]
        self.initial_conditions = {T: self.param.T_init}

    def _surface_cooling_coefficient(self):
        """Returns the surface cooling coefficient in for x-lumped models."""
        # Account for surface area to volume ratio in cooling coefficient
        # Note: assumes pouch cell geometry
        A = self.param.l_y * self.param.l_z
        V = self.param.l * self.param.l_y * self.param.l_z
        return -2 * self.param.h * A / V / (self.param.delta ** 2)
