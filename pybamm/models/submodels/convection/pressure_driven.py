#
# Class for pressure driven convection
#
import pybamm
from .base_convection import BaseModel


class PressureDriven(BaseModel):
    """Class for pressure-driven convection

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        v_box = pybamm.Scalar(0)

        variables = self._get_standard_velocity_variables(v_box)

        return variables
