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

    **Extends:** :class:`pybamm.convection.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

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

    def _get_standard_neg_pos_velocity_variables(self, v_box_n, v_box_p):
        """Volume-averaged velocity in the electrodes"""

        vel_scale = self.param.velocity_scale

        variables = {
            "Negative electrode volume-averaged velocity": v_box_n,
            "Positive electrode volume-averaged velocity": v_box_p,
            "Negative electrode volume-averaged velocity [m.s-1]": vel_scale * v_box_n,
            "Positive electrode volume-averaged velocity [m.s-1]": vel_scale * v_box_p,
        }

        return variables

    def _get_standard_neg_pos_acceleration_variables(self, div_v_box_n, div_v_box_p):
        """ Acceleration in the electrodes """

        acc_scale = self.param.velocity_scale / self.param.L_x

        div_v_box_n_av = pybamm.x_average(div_v_box_n)
        div_v_box_p_av = pybamm.x_average(div_v_box_p)

        variables = {
            "Negative electrode volume-averaged acceleration": div_v_box_n,
            "Positive electrode volume-averaged acceleration": div_v_box_p,
            "Negative electrode volume-averaged acceleration [m.s-1]": acc_scale
            * div_v_box_n,
            "Positive electrode volume-averaged acceleration [m.s-1]": acc_scale
            * div_v_box_p,
            "X-averaged negative electrode volume-averaged acceleration"
            + "": div_v_box_n_av,
            "X-averaged positive electrode volume-averaged acceleration"
            + "": div_v_box_p_av,
            "X-averaged negative electrode volume-averaged acceleration "
            + "[m.s-1]": acc_scale * div_v_box_n_av,
            "X-averaged positive electrode volume-averaged acceleration "
            + "[m.s-1]": acc_scale * div_v_box_p_av,
        }

        return variables

    def _get_standard_neg_pos_pressure_variables(self, p_n, p_p):
        """ Pressure in the electrodes """

        variables = {
            "Negative electrode pressure": p_n,
            "Positive electrode pressure": p_p,
            "X-averaged negative electrode pressure": pybamm.x_average(p_n),
            "X-averaged positive electrode pressure": pybamm.x_average(p_p),
        }

        return variables
