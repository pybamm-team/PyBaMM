#
# Submodel for uniform convection in transverse directions
#
import pybamm
from .base_transverse_convection import BaseTransverseModel


class Uniform(BaseTransverseModel):
    """
    Submodel for uniform convection in transverse directions

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.convection.through_cell.BaseTransverseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        p_s = pybamm.PrimaryBroadcast(0, "current collector")
        variables = self._get_standard_separator_pressure_variables(p_s)

        return variables

    def get_coupled_variables(self, variables):

        # Set up
        param = self.param
        z = pybamm.standard_spatial_vars.z

        # Difference in negative and positive electrode velocities determines the
        # velocity in the separator
        i_boundary_cc = variables["Current collector current density"]
        v_box_n_right = param.beta_n * i_boundary_cc
        v_box_p_left = param.beta_p * i_boundary_cc
        d_vbox_s_dx = (v_box_p_left - v_box_n_right) / param.l_s

        # Simple formula for velocity in the separator
        div_Vbox_s = -d_vbox_s_dx
        variables.update(
            self._get_standard_transverse_velocity_variables(div_Vbox_s, "acceleration")
        )

        Vbox_s = pybamm.IndefiniteIntegral(div_Vbox_s, z)
        variables.update(
            self._get_standard_transverse_velocity_variables(Vbox_s, "velocity")
        )

        return variables
