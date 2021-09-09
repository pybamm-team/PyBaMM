#
# Class for no SEI
#
import pybamm
from .base_sei import BaseModel


class NoSEI(BaseModel):
    """
    Class for no SEI.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    **Extends:** :class:`pybamm.sei.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), "negative electrode", "current collector"
        )
        variables = self._get_standard_thickness_variables(zero, zero)
        variables.update(self._get_standard_concentration_variables(variables))
        variables.update(self._get_standard_reaction_variables(zero, zero))
        return variables
