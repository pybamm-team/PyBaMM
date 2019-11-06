#
# A class for full external thermal models
#
import pybamm

from ..base_thermal import BaseThermal


class Full(BaseThermal):
    """Class to link full external thermal submodels.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_external_variables(self):
        T = pybamm.standard_variables.T
        T_cn = pybamm.BoundaryValue(T, "left")
        T_cp = pybamm.BoundaryValue(T, "right")

        variables = self._get_standard_fundamental_variables(T, T_cn, T_cp)
        external_variables = [T]

        return variables, external_variables
