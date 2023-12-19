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
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def _get_standard_convection_variables(self, domain, v_box_k, div_v_box_k, p_k):
        Domain = domain.capitalize()

        div_v_box_k_av = pybamm.x_average(div_v_box_k)

        variables = {
            f"{Domain} volume-averaged velocity [m.s-1]": v_box_k,
            f"{Domain} volume-averaged acceleration [m.s-2]": div_v_box_k,
            f"X-averaged {domain} volume-averaged acceleration [m.s-2]": div_v_box_k_av,
            f"{Domain} pressure [Pa]": p_k,
            f"X-averaged {domain} pressure [Pa]": pybamm.x_average(p_k),
        }
        return variables

    def _get_standard_sep_velocity_variables(self, v_box_s, div_v_box_s):
        """Volume-averaged velocity in the separator"""

        div_v_box_s_av = pybamm.x_average(div_v_box_s)

        variables = {
            "Separator volume-averaged velocity [m.s-1]": v_box_s,
            "Separator volume-averaged acceleration [m.s-2]": div_v_box_s,
            "X-averaged separator volume-averaged acceleration [m.s-2]": div_v_box_s_av,
        }

        return variables
