#
# Test the base experiment class
#

import os
from datetime import datetime

import casadi
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

import pybamm


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
            TypeError, match=r"Operating conditions must be a Step object or string."
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
            TypeError, match=r"Operating conditions must be a Step object or string."
        ):
            pybamm.Experiment([1, 2, 3])
        with pytest.raises(
            TypeError, match=r"Operating conditions must be a Step object or string."
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

        with pytest.raises(ValueError, match=r"Only capacity"):
            experiment = pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds"], termination="bla bla capacity bla"
            )
        with pytest.raises(ValueError, match=r"Only capacity"):
            experiment = pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds"], termination="4 A.h something else"
            )
        with pytest.raises(ValueError, match=r"Capacity termination"):
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
        with pytest.raises(ValueError, match=r"first step must have a `start_time`"):
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
        for next, end, steps in zip(
            expected_next, expected_end, processed_steps, strict=False
        ):
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
            match=r"Termination must include an operator when using InputParameter.",
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

        solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)
        sim = pybamm.Simulation(model, experiment=experiment, solver=solver)

        sim.solve()
        solution = sim.solution

        voltage = solution["Terminal voltage [V]"].entries
        assert np.allclose(voltage, 2.5, atol=1e-3, rtol=1e-3)

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
        with pytest.raises(ValueError, match=r"strictly increasing sequence"):
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

    def test_spm_3d_vs_lumped_pouch(self):
        models = {
            "Lumped": pybamm.lithium_ion.SPM(options={"thermal": "lumped"}),
            "3D": pybamm.lithium_ion.Basic3DThermalSPM(
                options={"cell geometry": "pouch", "dimensionality": 3}
            ),
        }

        parameter_values = pybamm.ParameterValues("Marquis2019")
        h_values = [0.1, 1, 10, 100]

        experiment = pybamm.Experiment(
            [
                ("Discharge at 3C until 2.8V", "Rest for 10 minutes"),
            ]
        )

        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "x": None,
            "y": None,
            "z": None,
        }

        all_solutions = {}

        for h in h_values:
            h_params = parameter_values.copy()
            h_params.update(
                {
                    "Total heat transfer coefficient [W.m-2.K-1]": h,
                    "Left face heat transfer coefficient [W.m-2.K-1]": h,
                    "Right face heat transfer coefficient [W.m-2.K-1]": h,
                    "Front face heat transfer coefficient [W.m-2.K-1]": h,
                    "Back face heat transfer coefficient [W.m-2.K-1]": h,
                    "Bottom face heat transfer coefficient [W.m-2.K-1]": h,
                    "Top face heat transfer coefficient [W.m-2.K-1]": h,
                },
                check_already_exists=False,
            )

            solutions = {}
            for model_name, model in models.items():
                sim = pybamm.Simulation(
                    model,
                    parameter_values=h_params,
                    var_pts=var_pts,
                    experiment=experiment,
                )
                solutions[model_name] = sim.solve()

            all_solutions[h] = solutions

        for _h, solutions in all_solutions.items():
            lumped_sol = solutions["Lumped"]
            three_d_sol = solutions["3D"]

            np.testing.assert_allclose(lumped_sol.t[-1], three_d_sol.t[-1], rtol=0.01)

            lumped_temp_final = lumped_sol[
                "Volume-averaged cell temperature [K]"
            ].entries[-1]
            three_d_temp_final = three_d_sol[
                "Volume-averaged cell temperature [K]"
            ].entries[-1]
            np.testing.assert_allclose(lumped_temp_final, three_d_temp_final, rtol=0.02)

            lumped_final_voltage = lumped_sol["Voltage [V]"].entries[-1]
            three_d_final_voltage = three_d_sol["Voltage [V]"].entries[-1]
            np.testing.assert_allclose(
                lumped_final_voltage, three_d_final_voltage, rtol=0.01
            )

    def test_spm_3d_vs_lumped_cylinder(self):
        models = {
            "Lumped": pybamm.lithium_ion.SPM(options={"thermal": "lumped"}),
            "3D": pybamm.lithium_ion.Basic3DThermalSPM(
                options={"cell geometry": "cylindrical", "dimensionality": 3}
            ),
        }

        parameter_values = pybamm.ParameterValues("NCA_Kim2011")
        h_values = [0.1, 1, 10, 100]

        experiment = pybamm.Experiment(
            [
                ("Discharge at 3C until 2.8V", "Rest for 10 minutes"),
            ]
        )

        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "r_macro": None,
            "y": None,
            "z": None,
        }

        all_solutions = {}

        for h in h_values:
            h_params = parameter_values.copy()
            h_params.update(
                {
                    "Inner cell radius [m]": 0.005,
                    "Outer cell radius [m]": 0.018,
                    "Total heat transfer coefficient [W.m-2.K-1]": h,
                    "Outer radius heat transfer coefficient [W.m-2.K-1]": h,
                    "Inner radius heat transfer coefficient [W.m-2.K-1]": h,
                    "Bottom face heat transfer coefficient [W.m-2.K-1]": h,
                    "Top face heat transfer coefficient [W.m-2.K-1]": h,
                },
                check_already_exists=False,
            )

            solutions = {}
            for model_name, model in models.items():
                sim = pybamm.Simulation(
                    model,
                    parameter_values=h_params,
                    var_pts=var_pts,
                    experiment=experiment,
                )
                solutions[model_name] = sim.solve()

            all_solutions[h] = solutions

        for _h, solutions in all_solutions.items():
            lumped_sol = solutions["Lumped"]
            three_d_sol = solutions["3D"]

            np.testing.assert_allclose(lumped_sol.t[-1], three_d_sol.t[-1], rtol=0.01)
            lumped_temp_final = lumped_sol[
                "Volume-averaged cell temperature [K]"
            ].entries[-1]
            three_d_temp_final = three_d_sol[
                "Volume-averaged cell temperature [K]"
            ].entries[-1]
            np.testing.assert_allclose(lumped_temp_final, three_d_temp_final, rtol=0.02)

            lumped_final_voltage = lumped_sol["Voltage [V]"].entries[-1]
            three_d_final_voltage = three_d_sol["Voltage [V]"].entries[-1]
            np.testing.assert_allclose(
                lumped_final_voltage, three_d_final_voltage, rtol=0.01
            )

    def test_spm_3d_vs_lumped_cylinder_with_custom_mesh(self):
        models = {
            "Lumped": pybamm.lithium_ion.SPM(options={"thermal": "lumped"}),
            "3D": pybamm.lithium_ion.Basic3DThermalSPM(
                options={"cell geometry": "cylindrical", "dimensionality": 3}
            ),
        }

        boundary_mapping = {"r_min": 1, "r_max": 2, "z_min": 3, "z_max": 4}
        domain_mapping = {"current collector": 5}

        MESH_DIR = os.path.join(os.path.dirname(__file__), "assets")
        file_path = os.path.join(MESH_DIR, "spm_test_mesh.msh")
        mesh_generator = pybamm.UserSuppliedSubmesh3D(
            file_path=file_path,
            boundary_mapping=boundary_mapping,
            domain_mapping=domain_mapping,
            coord_sys="cylindrical polar",
        )

        parameter_values = pybamm.ParameterValues("NCA_Kim2011")
        h = 10

        parameter_values.update(
            {
                "Inner cell radius [m]": 0.005,
                "Outer cell radius [m]": 0.018,
                "Cell height [m]": 0.065,
                "Total heat transfer coefficient [W.m-2.K-1]": h,
                "Outer radius heat transfer coefficient [W.m-2.K-1]": h,
                "Inner radius heat transfer coefficient [W.m-2.K-1]": h,
                "Bottom face heat transfer coefficient [W.m-2.K-1]": h,
                "Top face heat transfer coefficient [W.m-2.K-1]": h,
            },
            check_already_exists=False,
        )

        experiment = pybamm.Experiment(
            [("Discharge at 3C until 2.8V", "Rest for 10 minutes")]
        )

        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "r_macro": None,
            "y": None,
            "z": None,
        }

        solutions = {}
        for model_name, model in models.items():
            if model_name == "3D":
                submesh_types = model.default_submesh_types
                submesh_types["cell"] = mesh_generator

                sim = pybamm.Simulation(
                    model,
                    parameter_values=parameter_values,
                    var_pts=var_pts,
                    experiment=experiment,
                    submesh_types=submesh_types,
                )
            else:
                sim = pybamm.Simulation(
                    model,
                    parameter_values=parameter_values,
                    var_pts=var_pts,
                    experiment=experiment,
                )
            solutions[model_name] = sim.solve()

        lumped_sol = solutions["Lumped"]
        three_d_sol = solutions["3D"]

        lumped_temp = lumped_sol["Volume-averaged cell temperature [K]"].entries[-1]
        three_d_temp = three_d_sol["Volume-averaged cell temperature [K]"].entries[-1]
        np.testing.assert_allclose(lumped_temp, three_d_temp, rtol=0.02)

        lumped_voltage = lumped_sol["Voltage [V]"].entries[-1]
        three_d_voltage = three_d_sol["Voltage [V]"].entries[-1]
        np.testing.assert_allclose(lumped_voltage, three_d_voltage, rtol=0.01)
