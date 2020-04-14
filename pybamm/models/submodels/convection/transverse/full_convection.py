#
# Submodel for pressure driven convection in transverse directions
#
import pybamm
from .base_transverse_convection import BaseTransverseModel


class Full(BaseTransverseModel):
    """
    Submodel for the full model of pressure-driven convection in transverse directions

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    **Extends:** :class:`pybamm.convection.through_cell.BaseTransverseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        p_s = pybamm.Variable("Separator pressure", domain="current collector")
        variables = self._get_standard_separator_pressure_variables(p_s)

        # TODO: put in permeability and viscosity
        Vbox_s = -pybamm.grad(p_s)
        variables.update(
            self._get_standard_transverse_velocity_variables(Vbox_s, "velocity")
        )

        div_Vbox_s = pybamm.div(Vbox_s)
        variables.update(
            self._get_standard_transverse_velocity_variables(div_Vbox_s, "acceleration")
        )

        return variables

    def set_algebraic(self, variables):
        param = self.param

        p_s = variables["X-averaged separator pressure"]
        # Difference in negative and positive electrode velocities determines the
        # velocity in the separator
        i_boundary_cc = variables["Current collector current density"]
        v_box_n_right = param.beta_n * i_boundary_cc
        v_box_p_left = param.beta_p * i_boundary_cc
        d_vbox_s_dx = (v_box_p_left - v_box_n_right) / param.l_s

        # Simple formula for velocity in the separator
        div_Vbox_s = -d_vbox_s_dx
        Vbox_s = variables["X-averaged separator transverse volume-averaged velocity"]

        # Problem in the z-direction for p_s
        self.algebraic = {p_s: pybamm.div(Vbox_s) - div_Vbox_s}

    def set_boundary_conditions(self, variables):
        p_s = variables["X-averaged separator pressure"]

        # Boundary conditions in z-direction for p_s (left=bottom, right=top)
        self.boundary_conditions = {
            p_s: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Dirichlet"),
            }
        }

    def set_initial_conditions(self, variables):
        p_s = variables["X-averaged separator pressure"]

        self.initial_conditions = {p_s: pybamm.Scalar(0)}
