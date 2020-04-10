#
# Class for leading-order pressure driven convection
#
import pybamm
from .base_through_cell_convection import BaseThroughCellModel


class Explicit(BaseThroughCellModel):
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

        # Set up
        param = self.param
        l_n = pybamm.geometric_parameters.l_n
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        p_s = variables["X-averaged separator pressure"]
        j_n_av = variables["X-averaged negative electrode interfacial current density"]
        j_p_av = variables["X-averaged positive electrode interfacial current density"]

        p_n = param.beta_n * j_n_av * (-(x_n ** 2) + param.l_n ** 2) / 2 + p_s
        p_p = param.beta_n * j_n_av * ((x_p - 1) ** 2 - param.l_p ** 2) / 2 + p_s
        variables.update(self._get_standard_neg_pos_pressure_variables(p_n, p_p))

        # Volume-averaged velocity
        v_box_n = param.beta_n * j_n_av * x_n
        v_box_p = param.beta_p * j_p_av * (x_p - 1)
        variables.update(
            self._get_standard_neg_pos_velocity_variables(v_box_n, v_box_p)
        )

        div_v_box_n = pybamm.PrimaryBroadcast(
            param.beta_n * j_n_av, "negative electrode"
        )
        div_v_box_p = pybamm.PrimaryBroadcast(
            param.beta_p * j_p_av, "positive electrode"
        )
        variables.update(
            self._get_standard_neg_pos_acceleration_variables(div_v_box_n, div_v_box_p)
        )

        # Transverse velocity in the separator determines through-cell velocity
        div_Vbox_s = variables[
            "X-averaged separator transverse volume-averaged acceleration"
        ]
        i_boundary_cc = variables["Current collector current density"]
        v_box_n_right = param.beta_n * i_boundary_cc
        div_v_box_s_av = -div_Vbox_s
        div_v_box_s = pybamm.PrimaryBroadcast(div_v_box_s_av, "separator")

        # Simple formula for velocity in the separator
        v_box_s = div_v_box_s_av * (x_s - l_n) + v_box_n_right

        variables.update(
            self._get_standard_sep_velocity_variables(v_box_s, div_v_box_s)
        )
        variables.update(self._get_standard_whole_cell_velocity_variables(variables))
        variables.update(
            self._get_standard_whole_cell_acceleration_variables(variables)
        )
        variables.update(self._get_standard_whole_cell_pressure_variables(variables))

        return variables
