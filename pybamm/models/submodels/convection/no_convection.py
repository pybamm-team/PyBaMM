#
# No convection
#
import pybamm
from .base_convection import BaseModel


class NoConvection(BaseModel):
    """A submodel for case where there is no convection.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.convection.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        v_box_n = pybamm.FullBroadcast(0, "negative electrode", "current collector")
        v_box_s = pybamm.FullBroadcast(0, "separator", "current collector")
        v_box_p = pybamm.FullBroadcast(0, "positive electrode", "current collector")
        v_box = pybamm.Concatenation(v_box_n, v_box_s, v_box_p)
        variables = self._get_standard_velocity_variables(v_box)

        div_v_box_n = pybamm.FullBroadcast(0, "negative electrode", "current collector")
        div_v_box_s = pybamm.FullBroadcast(0, "separator", "current collector")
        div_v_box_p = pybamm.FullBroadcast(0, "positive electrode", "current collector")
        div_v_box = pybamm.Concatenation(div_v_box_n, div_v_box_s, div_v_box_p)
        variables = self._get_standard_acceleration_variables(div_v_box)

        p_n = pybamm.FullBroadcast(0, "negative electrode", "current collector")
        p_s = pybamm.FullBroadcast(0, "separator", "current collector")
        p_p = pybamm.FullBroadcast(0, "positive electrode", "current collector")
        p = pybamm.Concatenation(p_n, p_s, p_p)
        variables.update(self._get_standard_pressure_variables(p))

        return variables
