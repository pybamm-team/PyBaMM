#
# External circuit with power control
#
import pybamm
from .function_control_external_circuit import FunctionControl


class PowerControl(FunctionControl):
    """External circuit with power control. """

    def __init__(self, param):
        super().__init__(param, ConstantPower())


class ConstantPower:
    num_switches = 0

    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return I * V - pybamm.FunctionParameter("Power function", pybamm.t)

