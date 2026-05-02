"""
Regression tests for historical solver bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

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
        (i.e., exactly at the threshold), it wouldn't be caught.

        The fix changed:
        - if any(events_eval < 0):
        + if events_eval.min() <= 0:
        """
        param = pybamm.ParameterValues("Chen2020")

        model_custom = pybamm.lithium_ion.SPM()

        time = pybamm.Time()
        zero_event = pybamm.Event(
            "Zero at start",
            -time,
            pybamm.EventType.TERMINATION,
        )
        model_custom.events.append(zero_event)

        sim_custom = pybamm.Simulation(model_custom, parameter_values=param)

        with pytest.raises(pybamm.SolverError, match="non-positive at initial"):
            sim_custom.solve([0, 100])


class TestTimeIntegralFixes:
    """Guards for time integral / processed variable fixes."""

    def test_time_integral_with_input_parameter(self):
        """
        Guards against: PR #5120 - fix: misaligned args for post_sum_node
        evaluate

        The bug was that the evaluate method call was missing the `None`
        argument for the `u` parameter:
        - self.post_sum_node.evaluate(0.0, the_integral, inputs)
        + self.post_sum_node.evaluate(0.0, the_integral, None, inputs)
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        param.update({"Current function [A]": "[input]"})

        sim = pybamm.Simulation(model, parameter_values=param)

        sol = sim.solve(
            [0, 600],
            inputs={"Current function [A]": 5.0},
        )

        Q = sol["Discharge capacity [A.h]"].data
        assert not np.any(np.isnan(Q))
        assert Q[-1] > Q[0]

        throughput = sol["Throughput capacity [A.h]"].data
        assert not np.any(np.isnan(throughput))
        assert throughput[-1] > throughput[0]
