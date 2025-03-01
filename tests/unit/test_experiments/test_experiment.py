#
# Test the base experiment class
#

from datetime import datetime
import pybamm
import pytest
import numpy as np
from pybamm.experiment.step.base_step import process_temperature_input
import casadi
from scipy.interpolate import PchipInterpolator


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

    def test_temperature_time_series_simulation_build(self):
        time_data = np.array([0, 600, 1200, 1800])
        voltage_data = np.array([4.2, 4.0, 3.8, 3.6])
        temperature_data = np.array([298.15, 310.15, 305.15, 300.00])

        voltage_profile = np.column_stack((time_data, voltage_data))
        temperature_profile = np.column_stack((time_data, temperature_data))

        experiment = pybamm.Experiment(
            [pybamm.step.voltage(voltage_profile, temperature=temperature_profile)]
        )

        model = pybamm.lithium_ion.DFN()

        param_values = pybamm.ParameterValues("Marquis2019")
        param_values.update({"Ambient temperature [K]": 298.15})

        sim = pybamm.Simulation(
            model, experiment=experiment, parameter_values=param_values
        )
        sim.build()

        ambient_temp = sim.parameter_values["Ambient temperature [K]"]
        assert hasattr(ambient_temp, "evaluate"), (
            "Ambient temperature parameter is not time-dependent as expected."
        )

        t_eval = 600
        interpolated_temp = ambient_temp.evaluate(t=t_eval)
        np.testing.assert_allclose(interpolated_temp, 310.15, atol=1e-3)

        t_eval2 = 1200
        interpolated_temp2 = ambient_temp.evaluate(t=t_eval2)
        np.testing.assert_allclose(interpolated_temp2, 305.15, atol=1e-3)

    def test_process_temperature_valid(self):
        time_data = np.array([0, 600, 1200, 1800])
        temperature_data = np.array([298.15, 310.15, 305.15, 300.00])
        temperature_profile = np.column_stack((time_data, temperature_data))

        result = process_temperature_input(temperature_profile)

        assert isinstance(result, pybamm.Interpolant), "Expected an Interpolant object"
        np.testing.assert_allclose(result.evaluate(600), 310.15, atol=1e-3)
        np.testing.assert_allclose(result.evaluate(1200), 305.15, atol=1e-3)

    def test_process_temperature_invalid(self):
        invalid_data = np.array([298.15, 310.15, 305.15, 300.00])
        with pytest.raises(
            ValueError, match="Temperature time-series must be a 2D array"
        ):
            process_temperature_input(invalid_data)

        invalid_data_2 = np.array([[0, 298.15, 310.15], [600, 305.15, 300.00]])
        with pytest.raises(
            ValueError, match="Temperature time-series must be a 2D array"
        ):
            process_temperature_input(invalid_data_2)

    def test_temperature_time_series_simulation_solve(self):
        time_data = np.array([0, 600, 1200, 1800])
        voltage_data = np.array([4.2, 4.0, 3.8, 3.6])
        temperature_data = np.array([298.15, 310.15, 305.15, 300.00])

        voltage_profile = np.column_stack((time_data, voltage_data))
        temperature_profile = np.column_stack((time_data, temperature_data))

        # Update experiment step to handle temperature profile as expected
        experiment = pybamm.Experiment(
            [
                pybamm.step.voltage(
                    voltage_profile, temperature=temperature_profile, duration=1800
                )
            ]
        )

        model = pybamm.lithium_ion.DFN()

        param_values = pybamm.ParameterValues("Marquis2019")

        sim = pybamm.Simulation(
            model, experiment=experiment, parameter_values=param_values
        )

        solution = sim.solve()

        ambient_temp = sim.parameter_values["Ambient temperature [K]"]
        assert hasattr(ambient_temp, "evaluate"), (
            "Ambient temperature parameter is not time-dependent as expected."
        )

        assert solution is not None, "Solution object is None."
        assert hasattr(solution, "t"), "Solution does not contain time vector."

        np.testing.assert_allclose(solution.t[0], time_data[0], atol=1e-3)

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
        x = np.linspace(0, 1, 11)
        y_values = x**3

        y = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(x, y_values, y, interpolator="pchip")

        test_points = np.linspace(0, 1, 21)
        casadi_y = casadi.MX.sym("y", len(test_points), 1)
        interp_casadi = interp.to_casadi(y=casadi_y)
        f = casadi.Function("f", [casadi_y], [interp_casadi])

        casadi_results = f(test_points.reshape((-1, 1)))
        expected = interp.evaluate(y=test_points)
        np.testing.assert_allclose(casadi_results, expected, rtol=1e-7, atol=1e-6)

    def test_pchip_interpolation_uniform_grid(self):
        x = np.linspace(0, 1, 11)
        y_values = np.sin(x)

        state = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(x, y_values, state, interpolator="pchip")

        test_points = np.linspace(0, 1, 21)
        expected = PchipInterpolator(x, y_values)(test_points)

        casadi_y = casadi.MX.sym("y", 1)
        interp_casadi = interp.to_casadi(y=casadi_y)
        f = casadi.Function("f", [casadi_y], [interp_casadi])
        result = np.array(f(test_points)).flatten()

        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-6)

    def test_pchip_interpolation_nonuniform_grid(self):
        x = np.array([0, 0.05, 0.2, 0.4, 0.65, 1.0])
        y_values = np.exp(-x)
        state = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(x, y_values, state, interpolator="pchip")

        test_points = np.linspace(0, 1, 21)
        expected = PchipInterpolator(x, y_values)(test_points)

        casadi_y = casadi.MX.sym("y", 1)
        interp_casadi = interp.to_casadi(y=casadi_y)
        f = casadi.Function("f", [casadi_y], [interp_casadi])
        result = np.array(f(test_points)).flatten()

        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-6)

    def test_pchip_non_increasing_x(self):
        x = np.array([0, 0.5, 0.5, 1.0])
        y_values = np.linspace(0, 1, 4)
        state = pybamm.StateVector(slice(0, 1))
        with pytest.raises(ValueError, match="strictly increasing sequence"):
            _ = pybamm.Interpolant(x, y_values, state, interpolator="pchip")

    def test_pchip_extrapolation(self):
        x = np.linspace(0, 1, 11)
        y_values = np.log1p(x)  # a smooth function on [0,1]
        state = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(x, y_values, state, interpolator="pchip")

        test_points = np.array([-0.1, 1.1])
        expected = PchipInterpolator(x, y_values)(test_points)

        casadi_y = casadi.MX.sym("y", 1)
        interp_casadi = interp.to_casadi(y=casadi_y)
        f = casadi.Function("f", [casadi_y], [interp_casadi])
        result = np.array(f(test_points)).flatten()

        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-6)
