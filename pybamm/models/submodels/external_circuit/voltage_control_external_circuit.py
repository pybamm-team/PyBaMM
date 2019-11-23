#
# External circuit with voltage control
#
import pybamm
from .function_control_external_circuit import FunctionControl


class VoltageControl(FunctionControl):
    """External circuit with voltage control. """

    def __init__(self, param):
        super().__init__(param, ConstantVoltage())


class ConstantVoltage:
    num_switches = 0

    def __call__(self, variables):
        V = variables["Terminal voltage [V]"]
        return V - pybamm.FunctionParameter("Voltage function", pybamm.t)

