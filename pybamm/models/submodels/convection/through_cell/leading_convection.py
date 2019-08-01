#
# Class for leading-order pressure driven convection
#
import pybamm
from .base_through_cell_convection import BaseThroughCellModel


class LeadingOrder(BaseThroughCellModel):
    """A submodel for the leading-order approximation of pressure-driven convection

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.convection.through_cell.BaseThroughCellModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_coupled_variables(self, variables):

        param = self.param
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        j_n_av = variables["X-averaged negative electrode interfacial current density"]
        j_p_av = variables["X-averaged positive electrode interfacial current density"]

        # Volume-averaged velocity
        v_box_n = param.beta_n * pybamm.outer(j_n_av, x_n)
        v_box_p = param.beta_p * pybamm.outer(j_p_av, x_p - 1)
        variables.update(
            self._get_standard_neg_pos_velocity_variables(v_box_n, v_box_p)
        )

        div_v_box_n = param.beta_n * j_n_av
        div_v_box_p = param.beta_p * j_p_av
        variables.update(
            self._get_standard_neg_pos_acceleration_variables(div_v_box_n, div_v_box_p)
        )

        return variables
