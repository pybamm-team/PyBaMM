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
    """

    def __init__(self, param):
        super().__init__(param)

    def build(self):
        p_s = pybamm.Variable(
            "X-averaged separator pressure [Pa]", domain="current collector"
        )
        self.variables.update(self._get_standard_separator_pressure_variables(p_s))

        Vbox_s = -pybamm.grad(p_s)
        self.variables.update(
            self._get_standard_transverse_velocity_variables(Vbox_s, "velocity")
        )

        div_Vbox_s = pybamm.div(Vbox_s)
        self.variables.update(
            self._get_standard_transverse_velocity_variables(div_Vbox_s, "acceleration")
        )

        # Problem in the z-direction for p_s
        self.algebraic[p_s] = div_Vbox_s - self._get_separator_acceleration()
        # left=bottom, right=top
        self.boundary_conditions[p_s] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Dirichlet"),
        }
        self.initial_conditions[p_s] = pybamm.Scalar(0)
