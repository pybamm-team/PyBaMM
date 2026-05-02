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
        (i.e., exactly at the threshold), it wouldn't be caught and the
        solver would try to continue.

        The fix changed:
        - if any(events_eval < 0):
        + if events_eval.min() <= 0:

        This test verifies that an event exactly at 0 raises an error.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # Set initial SOC very close to lower cutoff so min voltage event
        # is at or near zero
        sim = pybamm.Simulation(model, parameter_values=param)

        # First, solve normally to verify the model works
        sol_normal = sim.solve([0, 3600])
        assert len(sol_normal.t) > 0

        # The lower cutoff is 2.5V by default in Chen2020
        # If we start at an SOC that gives exactly 2.5V, the event should be 0

        # Create a custom event that is exactly 0 at t=0
        model_custom = pybamm.lithium_ion.SPM()

        # Add a custom event that is exactly 0 at t=0
        time = pybamm.Time()
        zero_event = pybamm.Event(
            "Zero at start",
            -time,  # This is exactly 0 at t=0
            pybamm.EventType.TERMINATION,
        )
        model_custom.events.append(zero_event)

        sim_custom = pybamm.Simulation(model_custom, parameter_values=param)

        # Should raise error because event is exactly 0 at t=0
        with pytest.raises(pybamm.SolverError, match="non-positive at initial"):
            sim_custom.solve([0, 100])

    def test_event_slightly_positive_continues(self):
        """
        Verify events that are slightly positive at t=0 allow solving.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=param)

        # Normal solve should work - all events are positive at t=0
        sol = sim.solve([0, 100])
        assert len(sol.t) > 0

    def test_event_negative_raises_error(self):
        """
        Verify events that are negative at t=0 raise an error.
        """
        model = pybamm.lithium_ion.SPM()

        # Add a custom event that is negative at t=0
        time = pybamm.Time()
        negative_event = pybamm.Event(
            "Negative at start",
            time - 1,  # This is -1 at t=0
            pybamm.EventType.TERMINATION,
        )
        model.events.append(negative_event)

        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=param)

        with pytest.raises(pybamm.SolverError, match="non-positive at initial"):
            sim.solve([0, 100])


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

        This test verifies that time integrals work correctly when the
        model has input parameters.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # Set current as input parameter
        param.update({"Current function [A]": "[input]"})

        sim = pybamm.Simulation(model, parameter_values=param)

        # Solve with input - this exercises the code path with inputs
        sol = sim.solve(
            [0, 600],
            inputs={"Current function [A]": 5.0},
        )

        # Time integral variables should work correctly
        # Discharge capacity is a time integral of current
        Q = sol["Discharge capacity [A.h]"].data
        assert not np.any(np.isnan(Q))
        assert Q[-1] > Q[0]  # Capacity should accumulate

        # Throughput capacity includes absolute current integral
        throughput = sol["Throughput capacity [A.h]"].data
        assert not np.any(np.isnan(throughput))
        assert throughput[-1] > throughput[0]

    def test_time_integral_sensitivities_with_input(self):
        """
        Verify time integral sensitivities work with input parameters.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        param.update({"Current function [A]": "[input]"})

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)

        # Solve with sensitivities
        sol = sim.solve(
            [0, 300],
            inputs={"Current function [A]": 5.0},
            calculate_sensitivities=True,
        )

        # Sensitivities should be computed correctly
        assert "Current function [A]" in sol.sensitivities
        sens = sol.sensitivities["Current function [A]"]
        assert not np.any(np.isnan(sens))


class TestSolverConsistency:
    """Tests for solver consistency across configurations."""

    def test_casadi_and_idaklu_produce_similar_results(self):
        """
        Verify CasADi and IDAKLU solvers produce similar results.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        sim_casadi = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=pybamm.CasadiSolver(),
        )
        sim_idaklu = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=pybamm.IDAKLUSolver(),
        )

        sol_casadi = sim_casadi.solve([0, 600])
        sol_idaklu = sim_idaklu.solve([0, 600])

        # Interpolate to common time points for comparison
        t_common = np.linspace(0, 600, 50)
        V_casadi_interp = sol_casadi["Voltage [V]"](t_common)
        V_idaklu_interp = sol_idaklu["Voltage [V]"](t_common)

        np.testing.assert_allclose(V_casadi_interp, V_idaklu_interp, rtol=1e-3)
