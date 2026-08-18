from unittest.mock import Mock

import numpy as np
import pytest

import pybamm


class TestInitialConditionEventFixes:
    """Guards for initial condition event violation bug fixes."""

    def test_event_exactly_zero_raises_error(self):
        """
        Guards against: PR #5260 - Fix bug with initial conditions that
        violate events

        The bug was that the check for event violation used `< 0` instead
        of `<= 0`. This meant if an event was exactly 0 at initial conditions
        (right at the boundary), it wasn't caught as a violation.

        The fix changed the check in base_solver.py from:
            if any(events_eval < 0):
        to:
            if events_eval.min() <= 0:

        This test creates an event `-time` which evaluates to exactly 0 at
        t=0, triggering the boundary condition that the old code missed.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        time = pybamm.Time()
        zero_event = pybamm.Event(
            "Zero at start",
            -time,
            pybamm.EventType.TERMINATION,
        )
        model.events.append(zero_event)

        sim = pybamm.Simulation(model, parameter_values=param)

        with pytest.raises(pybamm.SolverError, match="non-positive at initial"):
            sim.solve([0, 100])


class TestTimeIntegralFixes:
    """Guards for time integral / processed variable fixes."""

    def test_post_sum_node_evaluate_argument_order(self):
        """
        Guards against: PR #5120 - fix: misaligned args for post_sum_node
        evaluate

        The bug was in ProcessedVariableTimeIntegral.postfix() where
        post_sum_node.evaluate() was called with wrong argument order:
            self.post_sum_node.evaluate(0.0, the_integral, inputs)
        instead of:
            self.post_sum_node.evaluate(0.0, the_integral, None, inputs)

        The evaluate signature is (t, y, u, inputs), so passing `inputs`
        as the 3rd arg caused it to be treated as `u` (state vector).

        This test directly verifies the argument order by mocking
        post_sum_node and checking how evaluate() is called.
        """

        class MockPostSumNode:
            def __init__(self):
                self.call_args = None

            def evaluate(self, t, y, u, inputs):
                self.call_args = (t, y, u, inputs)
                return np.array([y * 2])

        mock_node = MockPostSumNode()
        time_integral = pybamm.ProcessedVariableTimeIntegral(
            method="continuous",
            sum_node=Mock(),
            initial_condition=0.0,
            discrete_times=None,
            post_sum_node=mock_node,
            post_sum=None,
        )

        entries = np.array([1.0, 2.0, 3.0])
        t_pts = np.array([0.0, 1.0, 2.0])
        inputs = {"Current function [A]": 5.0}

        time_integral.postfix(entries, t_pts, inputs)

        assert mock_node.call_args is not None
        t, _y, u, inp = mock_node.call_args
        assert t == 0.0
        assert u is None, "u argument should be None"
        assert inp == inputs, "inputs should be passed as 4th argument"
