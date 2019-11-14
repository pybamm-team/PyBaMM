#
# External circuit with voltage control
#
import pybamm
from .base_external_circuit import BaseModel


class VoltageControl(BaseModel):
    """External circuit with voltage control. """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        # Current is a variable
        param = self.param
        i_cell = pybamm.Variable("Total current density")
        I = i_cell * abs(param.I_typ)
        i_cell_dim = I / (param.n_electrodes_parallel * param.A_cc)

        variables = {
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
            "Current [A]": I,
        }

        return variables

    def set_initial_conditions(self, variables):
        # Initial condition as a guess for consistent initial conditions
        i_cell = variables["Total current density"]
        self.initial_conditions[i_cell] = self.param.current_with_time

    def set_algebraic(self, variables):
        # External circuit submodels are always equations on the current
        # Fix voltage to be equal to terminal voltage
        i_cell = variables["Total current density"]
        V = variables["Terminal voltage"]
        # TODO: find a way to get rid of 0 * i_cell
        self.algebraic[i_cell] = V - self.param.voltage_with_time + 0 * i_cell
