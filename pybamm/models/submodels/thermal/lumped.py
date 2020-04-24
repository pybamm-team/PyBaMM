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
        super().__init__(param, cc_dimension)

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
        # different cell geometries (see #718)
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
