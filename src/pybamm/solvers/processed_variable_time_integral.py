from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import casadi
import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid

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
            return np.array(
                [trapezoid(entries, t_pts, axis=0) + float(self.initial_condition)]
            )

    def postfix(self, entries, t_pts, inputs) -> np.ndarray:
        """
        Compute the postfix sum or integral of the entries over time points,
        """
        the_integral = self.postfix_sum(entries, t_pts)
        if self.post_sum_node is None:
            ret = the_integral
        elif self.post_sum is None:
            ret = self.post_sum_node.evaluate(0.0, the_integral, None, inputs).reshape(
                -1
            )
        else:
            ret = self.post_sum(0.0, the_integral, inputs).full().reshape(-1)
        return ret

    def postfix_sensitivities(
        self,
        var_name,
        entries,
        t_pts,
        inputs,
        sensitivities,
    ) -> np.ndarray:
        # post fix for discrete time integral won't give correct result
        # if ts are not equal to the discrete times. Raise error
        # in this case
        if self.method == "discrete":
            if not (
                len(t_pts) == len(self.discrete_times)
                and np.allclose(t_pts, self.discrete_times, atol=1e-10)
            ):
                raise pybamm.SolverError(
                    f'Processing discrete-time-sum variable "{var_name}": solution times '
                    "and discrete times of the time integral are not equal. Set 't_interp=discrete_sum_times' to "
                    f"ensure the correct times are used.\nSolution times: {t_pts}\nDiscrete Sum times: {self.discrete_times}"
                )

        the_integral = self.postfix_sum(sensitivities, t_pts)
        if self.post_sum_node is None:
            return the_integral

        y_casadi = casadi.MX.sym("y", entries.shape[0])
        sens_casadi = casadi.MX.sym("s_var", the_integral.shape)
        t_casadi = casadi.MX.sym("t")
        p_casadi = {
            name: casadi.MX.sym(
                name, 1 if not isinstance(value, np.ndarray) else value.shape[0]
            )
            for name, value in inputs.items()
        }
        p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])
        inputs_stacked = casadi.vertcat(*[v for v in inputs.values()])
        post_sum_casadi = self.post_sum_node.to_casadi(
            t_casadi, y_casadi, inputs=p_casadi
        )

        dpost_dy = casadi.jacobian(post_sum_casadi, y_casadi)
        dpost_dp = casadi.jacobian(post_sum_casadi, p_casadi_stacked)
        sens = dpost_dy @ sens_casadi + dpost_dp
        sens_fun = casadi.Function(
            "sens_fun",
            [t_casadi, y_casadi, p_casadi_stacked, sens_casadi],
            [sens],
        )
        sens_values = sens_fun(0.0, entries, inputs_stacked, the_integral)
        return sens_values.full()

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
            if isinstance(sum_node.initial_condition, pybamm.Symbol):
                initial_condition = sum_node.initial_condition.evaluate()
            else:
                initial_condition = sum_node.initial_condition
            return ProcessedVariableTimeIntegral(
                method="continuous",
                post_sum_node=post_sum_node,
                sum_node=sum_node,
                initial_condition=initial_condition,
                discrete_times=None,
            )
        else:
            raise ValueError("Unsupported variable type")  # pragma: no cover
