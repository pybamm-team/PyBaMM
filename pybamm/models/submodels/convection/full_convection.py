#
# Class for pressure driven convection
#
import pybamm
from .base_convection import BaseModel


class FullModel(BaseModel):
    """Class for pressure-driven convection

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        p = pybamm.standard_variables.pressure

        v_mass = -pybamm.grad(p)
        v_box = v_mass

        variables = self._get_standard_pressure_variables(p)
        variables.update(self._get_standard_velocity_variables(v_box))

        return variables

    def get_coupled_variables(self, variables):

        _, dVbox_dz = self._separator_velocities(variables)

        variables.update(self._get_standard_vertical_velocity_variables(dVbox_dz))

        return variables

    def set_algebraic(self, variables):
        p = variables["Electrolyte pressure"]
        j = variables["Interfacial current density"]
        v_box = variables["Volume-average velocity"]
        dVbox_dz = variables["Vertical volume-averaged acceleration"]

        self.algebraic = {p: pybamm.div(v_box) + dVbox_dz - self.param.beta * j}

    def set_boundary_conditions(self, variables):
        p = variables["Electrolyte pressure"]
        self.boundary_conditions = {
            p: {"left": (0, "Dirichlet"), "right": (0, "Neumann")}
        }

    def set_initial_conditions(self, variables):
        p = variables["Electrolyte pressure"]
        self.initial_conditions = {p: 0}
