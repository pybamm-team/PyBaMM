from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy.typing as npt
import numpy as np
import pybamm
import casadi


@dataclass
class ProcessedVariableTimeIntegral:
    method: Literal["discrete", "continuous"]
    post_sum_node: pybamm.Symbol
    post_sum_casadi: casadi.Function | None
    sum_node: pybamm.Symbol
    initial_condition: npt.NDArray[np.float64] | float
    discrete_times: npt.NDArray[np.float64] | None

    @staticmethod
    def to_post_sum_expr(
        var: pybamm.Symbol,
        integral_node: pybamm.ExplicitTimeIntegral | pybamm.DiscreteTimeSum,
    ) -> pybamm.Symbol:
        # Should not be any time or state vector nodes in the children
        if any(isinstance(c, (pybamm.Time, pybamm.StateVector)) for c in var.children):
            raise ValueError(
                "For expressions containing time integrals, "
                "time or state vector nodes should only appear within the "
                "time integral node. "
            )
        new_children = [
            pybamm.StateVector()
            if c == integral_node
            else ProcessedVariableTimeIntegral.to_post_sum_expr(c, integral_node)
            for c in var.children
        ]

        return var.create_copy(new_children)

    @staticmethod
    def from_pybamm_var(
        var: pybamm.Symbol,
    ) -> ProcessedVariableTimeIntegral | None:
        sum_node = None
        for symbol in var.pre_order():
            if isinstance(
                symbol, (pybamm.ExplicitTimeIntegral, pybamm.DiscreteTimeSum)
            ):
                if sum_node is not None:
                    sum_node = symbol
                else:
                    raise ValueError(
                        "More than one time integral node found in the variable"
                        f" {var.name}. Only one time integral node is allowed."
                    )
        if sum_node is None:
            return None
        post_sum_node = ProcessedVariableTimeIntegral.to_post_sum_expr(var, sum_node)
        if isinstance(sum_node, pybamm.DiscreteTimeSum):
            return ProcessedVariableTimeIntegral(
                method="discrete",
                post_sum_node=post_sum_node,
                sum_node=sum_node,
                post_sum_casadi=None,
                initial_condition=0.0,
                discrete_times=sum_node.sum_times,
            )
        elif isinstance(sum_node, pybamm.ExplicitTimeIntegral):
            return ProcessedVariableTimeIntegral(
                method="continuous",
                post_sum_node=post_sum_node,
                sum_node=sum_node,
                post_sum_casadi=None,
                initial_condition=sum_node.initial_condition.evaluate(),
                discrete_times=None,
            )
        else:
            raise ValueError("Unsupported variable type")  # pragma: no cover
