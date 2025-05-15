from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy.typing as npt
import numpy as np
import pybamm


@dataclass
class ProcessedVariableTimeIntegral:
    method: Literal["discrete", "continuous"]
    initial_condition: npt.NDArray[np.float64] | None | float
    discrete_times: npt.NDArray[np.float64] | None

    @staticmethod
    def from_pybamm_var(
        var: pybamm.DiscreteTimeSum | pybamm.ExplicitTimeIntegral,
    ) -> ProcessedVariableTimeIntegral:
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
