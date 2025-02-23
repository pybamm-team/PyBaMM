#
# Test the base experiment class
#

from datetime import datetime
import pybamm
import pytest
import numpy as np
from scipy.interpolate import PchipInterpolator
import casadi


class TestExperiment:
    def test_cycle_unpacking(self):
        experiment = pybamm.Experiment(
            [
                ("Discharge at C/20 for 0.5 hours", "Charge at C/5 for 45 minutes"),
                ("Discharge at C/20 for 0.5 hours"),
                "Charge at C/5 for 45 minutes",
            ]
        )
        assert [step.to_dict() for step in experiment.steps] == [
            {
                "value": 0.05,
                "type": "CRate",
                "duration": 1800.0,
                "period": None,
                "temperature": None,
                "description": "Discharge at C/20 for 0.5 hours",
                "termination": [],
                "tags": [],
                "start_time": None,
            },
            {
                "value": -0.2,
                "type": "CRate",
                "duration": 2700.0,
                "period": None,
                "temperature": None,
                "description": "Charge at C/5 for 45 minutes",
                "termination": [],
                "tags": [],
                "start_time": None,
            },
            {
                "value": 0.05,
                "type": "CRate",
                "duration": 1800.0,
                "period": None,
                "temperature": None,
                "description": "Discharge at C/20 for 0.5 hours",
                "termination": [],
                "tags": [],
                "start_time": None,
            },
            {
                "value": -0.2,
                "type": "CRate",
                "duration": 2700.0,
                "period": None,
                "temperature": None,
                "description": "Charge at C/5 for 45 minutes",
                "termination": [],
                "tags": [],
                "start_time": None,
            },
        ]
        assert experiment.cycle_lengths == [2, 1, 1]

    def test_invalid_step_type(self):
        unprocessed = {1.0}
        period = 1
        temperature = 300.0
        with pytest.raises(
            TypeError, match="Operating conditions must be a Step object or string."
        ):
            pybamm.Experiment.process_steps(unprocessed, period, temperature)

    def test_str_repr(self):
        conds = ["Discharge at 1 C for 20 seconds", "Charge at 0.5 W for 10 minutes"]
        experiment = pybamm.Experiment(conds)
        assert (
            str(experiment)
            == "[('Discharge at 1 C for 20 seconds',)"
            + ", ('Charge at 0.5 W for 10 minutes',)]"
        )
        assert (
            repr(experiment)
            == "pybamm.Experiment([('Discharge at 1 C for 20 seconds',)"
            + ", ('Charge at 0.5 W for 10 minutes',)])"
        )

    def test_bad_strings(self):
        with pytest.raises(
            TypeError, match="Operating conditions must be a Step object or string."
        ):
            pybamm.Experiment([1, 2, 3])
        with pytest.raises(
            TypeError, match="Operating conditions must be a Step object or string."
        ):
            pybamm.Experiment([(1, 2, 3)])

    def test_termination(self):
        experiment = pybamm.Experiment(["Discharge at 1 C for 20 seconds"])
        assert experiment.termination == {}

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination=["80.7% capacity"]
        )
        assert experiment.termination == {"capacity": (80.7, "%")}

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination=["80.7 % capacity"]
        )
        assert experiment.termination == {"capacity": (80.7, "%")}

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination=["4.1Ah capacity"]
        )
        assert experiment.termination == {"capacity": (4.1, "Ah")}

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination=["4.1 A.h capacity"]
        )
        assert experiment.termination == {"capacity": (4.1, "Ah")}

        with pytest.raises(ValueError, match="Only capacity"):
            experiment = pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds"], termination="bla bla capacity bla"
            )
        with pytest.raises(ValueError, match="Only capacity"):
            experiment = pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds"], termination="4 A.h something else"
            )
        with pytest.raises(ValueError, match="Capacity termination"):
            experiment = pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds"], termination="1 capacity"
            )

    def test_search_tag(self):
        s = pybamm.step.string
        experiment = pybamm.Experiment(
            [
                (s("Discharge at 1C for 0.5 hours", tags=["tag1"]),),
                s("Discharge at C/20 for 0.5 hours", tags=["tag2", "tag3"]),
                (
                    s("Charge at 0.5 C for 45 minutes", tags=["tag2"]),
                    s("Discharge at 1 A for 0.5 hours", tags=["tag3"]),
                ),
                s("Charge at 200 mA for 45 minutes", tags=["tag5"]),
                (
                    s("Discharge at 1W for 0.5 hours", tags=["tag4"]),
                    s("Charge at 200mW for 45 minutes", tags=["tag4"]),
                ),
                s("Rest for 10 minutes", tags=["tag1", "tag3", "tag4"]),
            ]
        )

        assert experiment.search_tag("tag1") == [0, 5]
        assert experiment.search_tag("tag2") == [1, 2]
        assert experiment.search_tag("tag3") == [1, 2, 5]
        assert experiment.search_tag("tag4") == [4, 5]
        assert experiment.search_tag("tag5") == [3]
        assert experiment.search_tag("no_tag") == []

    def test_no_initial_start_time(self):
        s = pybamm.step.string
        with pytest.raises(ValueError, match="first step must have a `start_time`"):
            pybamm.Experiment(
                [
                    s("Rest for 1 hour"),
                    s("Rest for 1 hour", start_time=datetime(2023, 1, 1, 8, 0)),
                ]
            )

    def test_set_next_start_time(self):
        raw_steps = [
            pybamm.step.Current(
                1, duration=3600, start_time=datetime(2023, 1, 1, 8, 0)
            ),
            pybamm.step.Voltage(2.5, duration=3600, start_time=None),
            pybamm.step.Current(
                1, duration=3600, start_time=datetime(2023, 1, 1, 12, 0)
            ),
            pybamm.step.Current(1, duration=3600, start_time=None),
            pybamm.step.Voltage(2.5, duration=3600, start_time=None),
            pybamm.step.Current(
                1, duration=3600, start_time=datetime(2023, 1, 1, 15, 0)
            ),
        ]
        experiment = pybamm.Experiment(raw_steps)
        processed_steps = experiment._set_next_start_time(raw_steps)

        expected_next = [
            None,
            datetime(2023, 1, 1, 12, 0),
            None,
            None,
            datetime(2023, 1, 1, 15, 0),
            None,
        ]

        expected_end = [
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 1, 15, 0),
            datetime(2023, 1, 1, 15, 0),
            datetime(2023, 1, 1, 15, 0),
            None,
        ]

        # Test method directly
        for next, end, steps in zip(expected_next, expected_end, processed_steps):
            # useful form for debugging
            assert steps.next_start_time == next
            assert steps.end_time == end

        # TODO: once #3176 is completed, the test should pass for
        # operating_conditions_steps (or equivalent) as well

    def test_simulation_solve_updates_input_parameters(self):
        model = pybamm.lithium_ion.SPM()

        step = pybamm.step.current(
            pybamm.InputParameter("I_app"),
            termination="< 2.5 V",
        )
        experiment = pybamm.Experiment([step])

        sim = pybamm.Simulation(model, experiment=experiment)

        sim.solve(inputs={"I_app": 1})
        solution = sim.solution

        current = solution["Current [A]"].entries

        assert np.allclose(current, 1, atol=1e-3)

    def test_current_step_raises_error_without_operator_with_input_parameters(self):
        pybamm.lithium_ion.SPM()
        with pytest.raises(
            ValueError,
            match="Termination must include an operator when using InputParameter.",
        ):
            pybamm.step.current(pybamm.InputParameter("I_app"), termination="2.5 V")

    def test_value_function_with_input_parameter(self):
        I_coeff = pybamm.InputParameter("I_coeff")
        t = pybamm.t
        expr = I_coeff * t
        step = pybamm.step.current(expr, termination="< 2.5V")

        direction = step.value_based_charge_or_discharge()
        assert direction is None, (
            "Expected direction to be None when the expression depends on an InputParameter."
        )

    def test_symbolic_current_step(self):
        model = pybamm.lithium_ion.SPM()
        expr = 2.5 + 0 * pybamm.t

        step = pybamm.step.current(expr, duration=3600)
        experiment = pybamm.Experiment([step])

        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve([0, 3600])

        solution = sim.solution
        voltage = solution["Current [A]"].entries

        np.testing.assert_allclose(voltage[-1], 2.5, atol=0.1)

    def test_voltage_without_directions(self):
        model = pybamm.lithium_ion.SPM()

        step = pybamm.step.voltage(2.5, termination="2.5 V")
        experiment = pybamm.Experiment([step])

        sim = pybamm.Simulation(model, experiment=experiment)

        sim.solve()
        solution = sim.solution

        voltage = solution["Terminal voltage [V]"].entries
        assert np.allclose(voltage, 2.5, atol=1e-3)

    def test_pchip_interpolation_experiment(self):
        # Create a 1D grid and a known monotonic function.
        x = np.linspace(0, 1, 11)
        # Let f(x) = x**3, which is smooth and monotonic.
        y_values = x**3

        # Create a PyBaMM state vector for a 1D problem.
        y = pybamm.StateVector(slice(0, 1))

        # Create a PCHIP interpolant using PyBaMM.
        interp = pybamm.Interpolant(x, y_values, y, interpolator="pchip")

        # Convert the PyBaMM interpolant to a CasADi function.
        casadi_y = casadi.MX.sym("y", 1)
        interp_casadi = interp.to_casadi(y=casadi_y)
        f = casadi.Function("f", [casadi_y], [interp_casadi])

        # Choose some test points for evaluation.
        test_points = np.linspace(0, 1, 21)

        # Compute the expected results using SciPy's PchipInterpolator.
        expected = PchipInterpolator(x, y_values)(test_points)
        casadi_results = f(test_points)

        # Compare the results.
        np.testing.assert_allclose(
            np.array(casadi_results).flatten(), expected.flatten(), rtol=1e-7, atol=1e-6
        )
