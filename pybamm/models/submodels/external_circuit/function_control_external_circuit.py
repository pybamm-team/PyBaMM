#
# External circuit with an arbitrary function
#
import pybamm
from .base_external_circuit import BaseModel, LeadingOrderBaseModel


class FunctionControl(BaseModel):
    """External circuit with an arbitrary function. """

    def __init__(self, param, external_circuit_class):
        super().__init__(param)
        self.external_circuit_class = external_circuit_class

    def _get_current_variable(self):
        return pybamm.Variable("Total current density")

    def get_fundamental_variables(self):
        # Current is a variable
        i_cell = self._get_current_variable()
        variables = self._get_current_variables(i_cell)

        # Add discharge capacity variable
        variables.update(super().get_fundamental_variables())

        # Add switches
        # These are not implemented yet but can be used later with the Experiment class
        # to simulate different external circuit conditions sequentially within a
        # single model (for example Constant Current - Constant Voltage)
        # for i in range(self.external_circuit_class.num_switches):
        #     s = pybamm.Parameter("Switch {}".format(i + 1))
        #     variables["Switch {}".format(i + 1)] = s

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
        self.algebraic[i_cell] = self.external_circuit_class(variables)


class VoltageFunctionControl(FunctionControl):
    """
    External circuit with voltage control, implemented as an extra algebraic equation.
    """

    def __init__(self, param):
        super().__init__(param, ConstantVoltage())


class ConstantVoltage:
    num_switches = 0

    def __call__(self, variables):
        V = variables["Terminal voltage [V]"]
        return V - pybamm.FunctionParameter("Voltage function [V]", pybamm.t)


class PowerFunctionControl(FunctionControl):
    """External circuit with power control. """

    def __init__(self, param):
        super().__init__(param, ConstantPower())


class ConstantPower:
    num_switches = 0

    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return I * V - pybamm.FunctionParameter("Power function [W]", pybamm.t)


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
        super().__init__(param, ConstantVoltage())


class LeadingOrderPowerFunctionControl(LeadingOrderFunctionControl):
    """External circuit with power control, at leading order. """

    def __init__(self, param):
        super().__init__(param, ConstantPower())

