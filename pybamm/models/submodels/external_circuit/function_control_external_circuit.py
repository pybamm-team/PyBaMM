#
# External circuit with an arbitrary function
#
import pybamm
from .base_external_circuit import BaseModel, LeadingOrderBaseModel


class FunctionControl(BaseModel):
    """External circuit with an arbitrary function. """

    def __init__(self, param, external_circuit_function):
        super().__init__(param)
        self.external_circuit_function = external_circuit_function

    def get_fundamental_variables(self):
        param = self.param
        # Current is a variable
        i_cell = pybamm.Variable("Total current density")

        # Update derived variables
        I = i_cell * pybamm.AbsoluteValue(param.I_typ)
        i_cell_dim = I / (param.n_electrodes_parallel * param.A_cc)

        variables = {
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
            "Current [A]": I,
            "C-rate": I / param.Q,
        }

        # Add discharge capacity variable
        variables.update(super().get_fundamental_variables())

        return variables

    def set_initial_conditions(self, variables):
        super().set_initial_conditions(variables)
        # Initial condition as a guess for consistent initial conditions
        i_cell = variables["Total current density"]
        self.initial_conditions[i_cell] = self.param.current_with_time

    def set_algebraic(self, variables):
        # External circuit submodels are always equations on the current
        # The external circuit function should fix either the current, or the voltage,
        # or a combination (e.g. I*V for power control)
        i_cell = variables["Total current density"]
        self.algebraic[i_cell] = self.external_circuit_function(variables)


class VoltageFunctionControl(FunctionControl):
    """
    External circuit with voltage control, implemented as an extra algebraic equation.
    """

    def __init__(self, param):
        super().__init__(param, self.constant_voltage)

    def constant_voltage(self, variables):
        V = variables["Terminal voltage [V]"]
        return V - pybamm.FunctionParameter(
            "Voltage function [V]", {"Time [s]": pybamm.t * self.param.timescale}
        )


class PowerFunctionControl(FunctionControl):
    """External circuit with power control. """

    def __init__(self, param):
        super().__init__(param, self.constant_power)

    def constant_power(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return I * V - pybamm.FunctionParameter(
            "Power function [W]", {"Time [s]": pybamm.t * self.param.timescale}
        )


class LeadingOrderFunctionControl(FunctionControl, LeadingOrderBaseModel):
    """External circuit with an arbitrary function, at leading order. """

    def __init__(self, param, external_circuit_class):
        super().__init__(param, external_circuit_class)

    def _get_current_variable(self):
        return pybamm.Variable("Leading-order total current density")


class LeadingOrderVoltageFunctionControl(LeadingOrderFunctionControl):
    """
    External circuit with voltage control, implemented as an extra algebraic equation,
    at leading order.
    """

    def __init__(self, param):
        super().__init__(param, self.constant_voltage)

    def constant_voltage(self, variables):
        V = variables["Terminal voltage [V]"]
        return V - pybamm.FunctionParameter(
            "Voltage function [V]", {"Time [s]": pybamm.t * self.param.timescale}
        )


class LeadingOrderPowerFunctionControl(LeadingOrderFunctionControl):
    """External circuit with power control, at leading order. """

    def __init__(self, param):
        super().__init__(param, self.constant_power)

    def constant_power(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return I * V - pybamm.FunctionParameter(
            "Power function [W]", {"Time [s]": pybamm.t * self.param.timescale}
        )
