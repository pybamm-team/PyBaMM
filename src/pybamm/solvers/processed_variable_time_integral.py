from dataclasses import dataclass
from typing import Literal, Optional, Union
import pybamm


@dataclass
class ProcessedVariableTimeIntegral:
    method: Literal["discrete", "continuous"]
    initial_condition: numpy.typing.NDArray
    discrete_times: Optional[numpy.typing.NDArray]

    @staticmethod
    def from_pybamm_var(
        var: Union[pybamm.DiscreteTimeSum, pybamm.ExplicitTimeIntegral],
    ) -> "ProcessedVariableTimeIntegral":
        if isinstance(var, pybamm.DiscreteTimeSum):
            return ProcessedVariableTimeIntegral(
                method="discrete", initial_condition=0.0, discrete_times=var.sum_times
            )
        elif isinstance(var, pybamm.ExplicitTimeIntegral):
            return ProcessedVariableTimeIntegral(
                method="continuous",
                initial_condition=var.initial_condition.evaluate(),
                discrete_times=None,
            )
        else:
            raise ValueError("Unsupported variable type")  # pragma: no cover
