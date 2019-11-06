#
# A class for external lumped thermal models
#
import pybamm

from ..base_thermal import BaseThermal


class Full(BaseThermal):
    """Class to link external lumped thermal submodels.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_external_variables(self):
        T_x_av = pybamm.standard_variables.T_av
        T_n = pybamm.PrimaryBroadcast(T_x_av, "negative electrode")
        T_s = pybamm.PrimaryBroadcast(T_x_av, "separator")
        T_p = pybamm.PrimaryBroadcast(T_x_av, "positive electrode")
        T = pybamm.Concatenation(T_n, T_s, T_p)

        T_cn = T_x_av
        T_cp = T_x_av

        variables = self._get_standard_fundamental_variables(T, T_cn, T_cp)
        variables = self._get_standard_fundamental_variables(T, T_cn, T_cp)

        external_variables = [T_x_av]

        return variables, external_variables
