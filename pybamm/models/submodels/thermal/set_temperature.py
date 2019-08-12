#
# Class for thermal submodel in which the temperature is set externally
#
import pybamm

from .base_thermal import BaseModel


class SetTemperature(BaseModel):
    """Class for lumped thermal submodel which *doesn't update the temperature

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
        T_av = variables["X-averaged cell temperature"]
        q = variables["Heat flux"]
        Q_av = variables["X-averaged total heating"]
        return T_av, q, Q_av

    def set_rhs(self, variables):
        T_av, _, _ = self._unpack(variables)

        # Dummy equation so that PyBaMM doesn't change the temperature during solve
        # i.e. d_T/d_t = 0. The (local) temperature is set externally between steps.
        self.rhs = {T_av: pybamm.Scalar(0)}
