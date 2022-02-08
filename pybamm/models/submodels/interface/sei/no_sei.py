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
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.sei.BaseModel`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)
        if self.half_cell:
            self.reaction_loc = "interface"
        else:
            self.reaction_loc = "full electrode"

    def get_fundamental_variables(self):
        if self.reaction_loc == "interface":
            zero = pybamm.PrimaryBroadcast(pybamm.Scalar(0), "current collector")
        else:
            zero = pybamm.FullBroadcast(
                pybamm.Scalar(0), "negative electrode", "current collector"
            )
        variables = self._get_standard_thickness_variables(zero, zero)
        variables.update(self._get_standard_concentration_variables(variables))
        variables.update(self._get_standard_reaction_variables(zero, zero))
        return variables
