#
# Base class for convection submodels in the through-cell direction
#
import pybamm
from ..base_convection import BaseModel


class BaseThroughCellModel(BaseModel):
    """Base class for convection submodels in the through-cell direction.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.convection.BaseModel`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def _get_standard_sep_velocity_variables(self, v_box_s, div_v_box_s):
        """Volume-averaged velocity in the separator"""

        vel_scale = self.param.velocity_scale
        div_v_box_s_av = pybamm.x_average(div_v_box_s)

        variables = {
            "Separator volume-averaged velocity": v_box_s,
            "Separator volume-averaged velocity [m.s-1]": vel_scale * v_box_s,
            "Separator volume-averaged acceleration": div_v_box_s,
            "Separator volume-averaged acceleration [m.s-1]": vel_scale * div_v_box_s,
            "X-averaged separator volume-averaged acceleration": div_v_box_s_av,
            "X-averaged separator volume-averaged acceleration "
            + "[m.s-1]": vel_scale * div_v_box_s_av,
        }

        return variables
