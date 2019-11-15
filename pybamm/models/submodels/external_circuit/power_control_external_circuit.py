#
# External circuit with power control
#
import pybamm
from .function_control_external_circuit import FunctionControl


class PowerControl(FunctionControl):
    """External circuit with power control. """

    def __init__(self, param):
        super().__init__(param)

    def external_circuit_function(self, I, V):
        return I * V - pybamm.FunctionParameter(
            "Power function", pybamm.t * self.param.timescale
        )

