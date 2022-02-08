#
# No convection
#
import pybamm
from .base_through_cell_convection import BaseThroughCellModel


class NoConvection(BaseThroughCellModel):
    """A submodel for case where there is no convection.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.convection.through_cell.BaseThroughCellModel`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def get_fundamental_variables(self):

        if self.half_cell:
            v_box_n = None
            div_v_box_n = None
            p_n = None
        else:
            v_box_n = pybamm.FullBroadcast(0, "negative electrode", "current collector")
            div_v_box_n = pybamm.FullBroadcast(
                0, "negative electrode", "current collector"
            )
            p_n = pybamm.FullBroadcast(0, "negative electrode", "current collector")
        v_box_p = pybamm.FullBroadcast(0, "positive electrode", "current collector")
        div_v_box_p = pybamm.FullBroadcast(0, "positive electrode", "current collector")
        p_p = pybamm.FullBroadcast(0, "positive electrode", "current collector")

        variables = self._get_standard_neg_pos_velocity_variables(v_box_n, v_box_p)
        variables.update(
            self._get_standard_neg_pos_acceleration_variables(div_v_box_n, div_v_box_p)
        )
        variables.update(self._get_standard_neg_pos_pressure_variables(p_n, p_p))

        return variables

    def get_coupled_variables(self, variables):

        # Simple formula for velocity in the separator
        v_box_s = pybamm.FullBroadcast(0, "separator", "current collector")
        div_v_box_s = pybamm.FullBroadcast(0, "separator", "current collector")

        variables.update(
            self._get_standard_sep_velocity_variables(v_box_s, div_v_box_s)
        )
        variables.update(self._get_standard_whole_cell_velocity_variables(variables))
        variables.update(
            self._get_standard_whole_cell_acceleration_variables(variables)
        )
        variables.update(self._get_standard_whole_cell_pressure_variables(variables))

        return variables
