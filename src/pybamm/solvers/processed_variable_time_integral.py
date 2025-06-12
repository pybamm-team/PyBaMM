from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import casadi
import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid

import pybamm


@dataclass
class ProcessedVariableTimeIntegral:
    method: Literal["discrete", "continuous"]
    sum_node: pybamm.Symbol
    initial_condition: npt.NDArray[np.float64] | float
    discrete_times: npt.NDArray[np.float64] | None
    post_sum_node: pybamm.Symbol | None = None
    post_sum: casadi.Function | None = None

    def postfix_sum(self, entries, t_pts) -> np.ndarray:
        if self.method == "discrete":
            return np.sum(
                entries, axis=0, initial=self.initial_condition, keepdims=True
            )
        else:
            return cumulative_trapezoid(
                entries, t_pts, initial=float(self.initial_condition)
            )

    def postfix(self, entries, t_pts, inputs) -> np.ndarray:
        """
        Compute the postfix sum or integral of the entries over time points,
        """
        the_integral = self.postfix_sum(entries, t_pts)
        if self.post_sum_node is None:
            return the_integral
        else:
            return self.post_sum(0.0, the_integral, inputs)

    @staticmethod
    def to_post_sum_expr(
        var: pybamm.Symbol,
        integral_node: pybamm.ExplicitTimeIntegral | pybamm.DiscreteTimeSum,
        nstates: int,
    ) -> pybamm.Symbol:
        if var == integral_node:
            # If the variable is the integral node itself, return a state vector
            return pybamm.StateVector(slice(0, nstates))

        # Should not be any time or state vector nodes in the children
        if any(isinstance(c, pybamm.Time | pybamm.StateVector) for c in var.children):
            raise ValueError(
                "For expressions containing time integrals, "
                "time or state vector nodes should only appear within the "
                "time integral node. "
            )
        new_children = [
            ProcessedVariableTimeIntegral.to_post_sum_expr(c, integral_node, nstates)
            for c in var.children
        ]
        return var.create_copy(new_children)

    @staticmethod
    def from_pybamm_var(
        var: pybamm.Symbol,
        nstates: int,
    ) -> ProcessedVariableTimeIntegral | None:
        sum_node = None
        for symbol in var.pre_order():
            if isinstance(symbol, pybamm.ExplicitTimeIntegral | pybamm.DiscreteTimeSum):
                if sum_node is None:
                    sum_node = symbol
                else:
                    raise ValueError(
                        "More than one time integral node found in the variable"
                        f" {var.name}. Only one time integral node is allowed."
                    )
        if sum_node is None:
            return None
        if sum_node == var:
            post_sum_node = None
        else:
            sum_y_len = sum_node.evaluate_for_shape().shape[0]
            post_sum_node = ProcessedVariableTimeIntegral.to_post_sum_expr(
                var, sum_node, sum_y_len
            )
        if isinstance(sum_node, pybamm.DiscreteTimeSum):
            return ProcessedVariableTimeIntegral(
                method="discrete",
                post_sum_node=post_sum_node,
                sum_node=sum_node,
                initial_condition=0.0,
                discrete_times=sum_node.sum_times,
            )
        elif isinstance(sum_node, pybamm.ExplicitTimeIntegral):
            return ProcessedVariableTimeIntegral(
                method="continuous",
                post_sum_node=post_sum_node,
                sum_node=sum_node,
                initial_condition=sum_node.initial_condition.evaluate(),
                discrete_times=None,
            )
        else:
            raise ValueError("Unsupported variable type")  # pragma: no cover
