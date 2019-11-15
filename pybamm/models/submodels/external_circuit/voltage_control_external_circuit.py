#
# External circuit with voltage control
#
import pybamm
from .function_control_external_circuit import FunctionControl


class VoltageControl(FunctionControl):
    """External circuit with voltage control. """

    def __init__(self, param):
        super().__init__(param)

    def external_circuit_function(self, I, V):
        return V - pybamm.FunctionParameter(
            "Voltage function", pybamm.t * self.param.timescale
        )

