#
# External circuit with an arbitrary function
#
import pybamm
from .base_external_circuit import BaseModel


class FunctionControl(BaseModel):
    """External circuit with an arbitrary function. """

    def __init__(self, param):
        super().__init__(param)

    def external_circuit_function(self, I, V):
        """
        Function that fixes the current, I (in Amps), or voltage, V (in Volts), or a
        combination of the two
        """
        return pybamm.FunctionParameter("External circuit function", I, V)

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
        # The external circuit function should fix either the current, or the voltage,
        # or a combination (e.g. I*V for power control)
        i_cell = variables["Total current density"]
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        self.algebraic[i_cell] = self.external_circuit_function(I, V)
