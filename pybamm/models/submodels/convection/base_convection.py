#
# Base class for convection
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for convection

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_velocity_variables(self, v_box):

        # add more to this
        variables = {"Volume-averaged velocity": v_box}

        return variables
