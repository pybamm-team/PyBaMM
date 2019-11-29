#
# External circuit with current control
#
import pybamm
from .base_external_circuit import BaseModel


class CurrentControl(BaseModel):
    """External circuit with current control. """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        # Current is given as a function of time
        i_cell = self.param.current_with_time
        i_cell_dim = self.param.dimensional_current_density_with_time
        I = self.param.dimensional_current_with_time

        variables = {
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
            "Current [A]": I,
        }

        # Add discharge capacity variable
        variables.update(super().get_fundamental_variables())

        return variables

    def get_coupled_variables(self, variables):
        # Update terminal voltage
        phi_s_cp_dim = variables["Positive current collector potential [V]"]
        phi_s_cp = variables["Positive current collector potential"]

        V = pybamm.boundary_value(phi_s_cp, "positive tab")
        V_dim = pybamm.boundary_value(phi_s_cp_dim, "positive tab")
        variables["Terminal voltage"] = V
        variables["Terminal voltage [V]"] = V_dim

        return variables
