#
# Tests for the Solution class
#
import os

import json
import pybamm
import unittest
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tests import get_discretisation_for_testing
from tempfile import TemporaryDirectory


class TestSolution(unittest.TestCase):
    def test_init(self):
        t = np.linspace(0, 1)
        y = np.tile(t, (20, 1))
        sol = pybamm.Solution(t, y, pybamm.BaseModel(), {})
        np.testing.assert_array_equal(sol.t, t)
        np.testing.assert_array_equal(sol.y, y)
        self.assertEqual(sol.t_event, None)
        self.assertEqual(sol.y_event, None)
        self.assertEqual(sol.termination, "final time")
        self.assertEqual(sol.all_inputs, [{}])
        self.assertIsInstance(sol.all_models[0], pybamm.BaseModel)

    def test_sensitivities(self):
        t = np.linspace(0, 1)
        y = np.tile(t, (20, 1))
        with self.assertRaises(TypeError):
            pybamm.Solution(t, y, pybamm.BaseModel(), {}, sensitivities=1.0)

    def test_errors(self):
        bad_ts = [np.array([1, 2, 3]), np.array([3, 4, 5])]
        sol = pybamm.Solution(
            bad_ts, [np.ones((1, 3)), np.ones((1, 3))], pybamm.BaseModel(), {}
        )
        with self.assertRaisesRegex(
            ValueError, "Solution time vector must be strictly increasing"
        ):
            sol.set_t()

        ts = [np.array([1, 2, 3])]
        bad_ys = [(pybamm.settings.max_y_value + 1) * np.ones((1, 3))]
        model = pybamm.BaseModel()
        var = pybamm.StateVector(slice(0, 1))
        model.rhs = {var: 0}
        model.variables = {var.name: var}
        with self.assertLogs() as captured:
            pybamm.Solution(ts, bad_ys, model, {})
        self.assertIn("exceeds the maximum", captured.records[0].getMessage())

    def test_add_solutions(self):
        # Set up first solution
        t1 = np.linspace(0, 1)
        y1 = np.tile(t1, (20, 1))
        sol1 = pybamm.Solution(t1, y1, pybamm.BaseModel(), {"a": 1})
        sol1.solve_time = 1.5
        sol1.integration_time = 0.3

        # Set up second solution
        t2 = np.linspace(1, 2)
        y2 = np.tile(t2, (20, 1))
        sol2 = pybamm.Solution(t2, y2, pybamm.BaseModel(), {"a": 2})
        sol2.solve_time = 1
        sol2.integration_time = 0.5

        sol_sum = sol1 + sol2

        # Test
        self.assertEqual(sol_sum.integration_time, 0.8)
        np.testing.assert_array_equal(sol_sum.t, np.concatenate([t1, t2[1:]]))
        np.testing.assert_array_equal(
            sol_sum.y, np.concatenate([y1, y2[:, 1:]], axis=1)
        )
        np.testing.assert_array_equal(sol_sum.all_inputs, [{"a": 1}, {"a": 2}])

        # Test sub-solutions
        self.assertEqual(len(sol_sum.sub_solutions), 2)
        np.testing.assert_array_equal(sol_sum.sub_solutions[0].t, t1)
        np.testing.assert_array_equal(sol_sum.sub_solutions[1].t, t2)
        self.assertEqual(sol_sum.sub_solutions[0].all_models[0], sol_sum.all_models[0])
        np.testing.assert_array_equal(sol_sum.sub_solutions[0].all_inputs[0]["a"], 1)
        self.assertEqual(sol_sum.sub_solutions[1].all_models[0], sol2.all_models[0])
        self.assertEqual(sol_sum.all_models[1], sol2.all_models[0])
        np.testing.assert_array_equal(sol_sum.sub_solutions[1].all_inputs[0]["a"], 2)

        # Add solution already contained in existing solution
        t3 = np.array([2])
        y3 = np.ones((20, 1))
        sol3 = pybamm.Solution(t3, y3, pybamm.BaseModel(), {"a": 3})
        self.assertEqual((sol_sum + sol3).all_ts, sol_sum.copy().all_ts)

        # add None
        sol4 = sol3 + None
        self.assertEqual(sol3.all_ys, sol4.all_ys)

        # radd
        sol5 = None + sol3
        self.assertEqual(sol3.all_ys, sol5.all_ys)

        # radd failure
        with self.assertRaisesRegex(
            pybamm.SolverError, "Only a Solution or None can be added to a Solution"
        ):
            sol3 + 2
        with self.assertRaisesRegex(
            pybamm.SolverError, "Only a Solution or None can be added to a Solution"
        ):
            2 + sol3

    def test_add_solutions_different_models(self):
        # Set up first solution
        t1 = np.linspace(0, 1)
        y1 = np.tile(t1, (20, 1))
        sol1 = pybamm.Solution(t1, y1, pybamm.BaseModel(), {"a": 1})
        sol1.solve_time = 1.5
        sol1.integration_time = 0.3

        # Set up second solution
        t2 = np.linspace(1, 2)
        y2 = np.tile(t2, (10, 1))
        sol2 = pybamm.Solution(t2, y2, pybamm.BaseModel(), {"a": 2})
        sol2.solve_time = 1
        sol2.integration_time = 0.5
        sol_sum = sol1 + sol2

        # Test
        np.testing.assert_array_equal(sol_sum.t, np.concatenate([t1, t2[1:]]))
        with self.assertRaisesRegex(
            pybamm.SolverError, "The solution is made up from different models"
        ):
            sol_sum.y

    def test_copy(self):
        # Set up first solution
        t1 = [np.linspace(0, 1), np.linspace(1, 2, 5)]
        y1 = [np.tile(t1[0], (20, 1)), np.tile(t1[1], (20, 1))]
        sol1 = pybamm.Solution(t1, y1, pybamm.BaseModel(), [{"a": 1}, {"a": 2}])

        sol1.set_up_time = 0.5
        sol1.solve_time = 1.5
        sol1.integration_time = 0.3

        sol_copy = sol1.copy()
        self.assertEqual(sol_copy.all_ts, sol1.all_ts)
        self.assertEqual(sol_copy.all_ys, sol1.all_ys)
        self.assertEqual(sol_copy.all_inputs, sol1.all_inputs)
        self.assertEqual(sol_copy.all_inputs_casadi, sol1.all_inputs_casadi)
        self.assertEqual(sol_copy.set_up_time, sol1.set_up_time)
        self.assertEqual(sol_copy.solve_time, sol1.solve_time)
        self.assertEqual(sol_copy.integration_time, sol1.integration_time)

    def test_last_state(self):
        # Set up first solution
        t1 = [np.linspace(0, 1), np.linspace(1, 2, 5)]
        y1 = [np.tile(t1[0], (20, 1)), np.tile(t1[1], (20, 1))]
        sol1 = pybamm.Solution(t1, y1, pybamm.BaseModel(), [{"a": 1}, {"a": 2}])

        sol1.set_up_time = 0.5
        sol1.solve_time = 1.5
        sol1.integration_time = 0.3

        sol_last_state = sol1.last_state
        self.assertEqual(sol_last_state.all_ts[0], 2)
        np.testing.assert_array_equal(sol_last_state.all_ys[0], 2)
        self.assertEqual(sol_last_state.all_inputs, sol1.all_inputs[-1:])
        self.assertEqual(sol_last_state.all_inputs_casadi, sol1.all_inputs_casadi[-1:])
        self.assertEqual(sol_last_state.all_models, sol1.all_models[-1:])
        self.assertEqual(sol_last_state.set_up_time, 0)
        self.assertEqual(sol_last_state.solve_time, 0)
        self.assertEqual(sol_last_state.integration_time, 0)

    def test_cycles(self):
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [
                ("Discharge at C/20 for 0.5 hours", "Charge at C/20 for 15 minutes"),
                ("Discharge at C/20 for 0.5 hours", "Charge at C/20 for 15 minutes"),
            ]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()
        self.assertEqual(len(sol.cycles), 2)
        len_cycle_1 = len(sol.cycles[0].t)

        self.assertIsInstance(sol.cycles[0], pybamm.Solution)
        np.testing.assert_array_equal(sol.cycles[0].t, sol.t[:len_cycle_1])
        np.testing.assert_array_equal(sol.cycles[0].y, sol.y[:, :len_cycle_1])

        self.assertIsInstance(sol.cycles[1], pybamm.Solution)
        np.testing.assert_array_equal(sol.cycles[1].t, sol.t[len_cycle_1:])
        np.testing.assert_allclose(sol.cycles[1].y, sol.y[:, len_cycle_1:])

    def test_total_time(self):
        sol = pybamm.Solution(np.array([0]), np.array([[1, 2]]), pybamm.BaseModel(), {})
        sol.set_up_time = 0.5
        sol.solve_time = 1.2
        self.assertEqual(sol.total_time, 1.7)

    def test_getitem(self):
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c
        model.variables["2c"] = 2 * c

        solution = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

        # test create a new processed variable
        c_sol = solution["c"]
        self.assertIsInstance(c_sol, pybamm.ProcessedVariable)
        np.testing.assert_array_equal(c_sol.entries, c_sol(solution.t))

        # test call an already created variable
        solution.update("2c")
        twoc_sol = solution["2c"]
        self.assertIsInstance(twoc_sol, pybamm.ProcessedVariable)
        np.testing.assert_array_equal(twoc_sol.entries, twoc_sol(solution.t))
        np.testing.assert_array_equal(twoc_sol.entries, 2 * c_sol.entries)

    def test_plot(self):
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c
        model.variables["2c"] = 2 * c

        solution = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

        solution.plot(["c", "2c"], show_plot=False)

    def test_save(self):
        with TemporaryDirectory() as dir_name:
            test_stub = os.path.join(dir_name, "test")

            model = pybamm.BaseModel()
            # create both 1D and 2D variables
            c = pybamm.Variable("c")
            d = pybamm.Variable("d", domain="negative electrode")
            model.rhs = {c: -c, d: 1}
            model.initial_conditions = {c: 1, d: 2}
            model.variables = {"c": c, "d": d, "2c": 2 * c, "c + d": c + d}

            disc = get_discretisation_for_testing()
            disc.process_model(model)
            solution = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

            # test save data
            with self.assertRaises(ValueError):
                solution.save_data(f"{test_stub}.pickle")

            # set variables first then save
            solution.update(["c", "d"])
            with self.assertRaisesRegex(ValueError, "pickle"):
                solution.save_data(to_format="pickle")
            solution.save_data(f"{test_stub}.pickle")

            data_load = pybamm.load(f"{test_stub}.pickle")
            np.testing.assert_array_equal(solution.data["c"], data_load["c"])
            np.testing.assert_array_equal(solution.data["d"], data_load["d"])

            # to matlab
            solution.save_data(f"{test_stub}.mat", to_format="matlab")
            data_load = loadmat(f"{test_stub}.mat")
            np.testing.assert_array_equal(solution.data["c"], data_load["c"].flatten())
            np.testing.assert_array_equal(solution.data["d"], data_load["d"])

            with self.assertRaisesRegex(ValueError, "matlab"):
                solution.save_data(to_format="matlab")

            # to matlab with bad variables name fails
            solution.update(["c + d"])
            with self.assertRaisesRegex(ValueError, "Invalid character"):
                solution.save_data(f"{test_stub}.mat", to_format="matlab")
            # Works if providing alternative name
            solution.save_data(
                f"{test_stub}.mat",
                to_format="matlab",
                short_names={"c + d": "c_plus_d"},
            )
            data_load = loadmat(f"{test_stub}.mat")
            np.testing.assert_array_equal(solution.data["c + d"], data_load["c_plus_d"])

            # to csv
            with self.assertRaisesRegex(
                ValueError, "only 0D variables can be saved to csv"
            ):
                solution.save_data(f"{test_stub}.csv", to_format="csv")
            # only save "c" and "2c"
            solution.save_data(f"{test_stub}.csv", ["c", "2c"], to_format="csv")
            csv_str = solution.save_data(variables=["c", "2c"], to_format="csv")

            # check string is the same as the file
            with open(f"{test_stub}.csv") as f:
                # need to strip \r chars for windows
                self.assertEqual(csv_str.replace("\r", ""), f.read())

            # read csv
            df = pd.read_csv(f"{test_stub}.csv")
            np.testing.assert_array_almost_equal(df["c"], solution.data["c"])
            np.testing.assert_array_almost_equal(df["2c"], solution.data["2c"])

            # to json
            solution.save_data(f"{test_stub}.json", to_format="json")
            json_str = solution.save_data(to_format="json")

            # check string is the same as the file
            with open(f"{test_stub}.json") as f:
                # need to strip \r chars for windows
                self.assertEqual(json_str.replace("\r", ""), f.read())

            # check if string has the right values
            json_data = json.loads(json_str)
            np.testing.assert_array_almost_equal(json_data["c"], solution.data["c"])
            np.testing.assert_array_almost_equal(json_data["d"], solution.data["d"])

            # raise error if format is unknown
            with self.assertRaisesRegex(
                ValueError, "format 'wrong_format' not recognised"
            ):
                solution.save_data(f"{test_stub}.csv", to_format="wrong_format")

            # test save whole solution
            solution.save(f"{test_stub}.pickle")
            solution_load = pybamm.load(f"{test_stub}.pickle")
            self.assertEqual(
                solution.all_models[0].name, solution_load.all_models[0].name
            )
            np.testing.assert_array_equal(
                solution["c"].entries, solution_load["c"].entries
            )
            np.testing.assert_array_equal(
                solution["d"].entries, solution_load["d"].entries
            )

    def test_get_data_cycles_steps(self):
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c

        solver = pybamm.ScipySolver()
        sol1 = solver.solve(model, np.linspace(0, 1))
        sol2 = solver.solve(model, np.linspace(1, 2))

        sol = sol1 + sol2
        sol.cycles = [sol]
        sol.cycles[0].steps = [sol1, sol2]

        data = sol.get_data_dict("c")
        np.testing.assert_array_equal(data["Cycle"], 0)
        np.testing.assert_array_equal(
            data["Step"], np.concatenate([np.zeros(50), np.ones(50)])
        )

    def test_solution_evals_with_inputs(self):
        model = pybamm.lithium_ion.SPM()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.update({"Negative electrode conductivity [S.m-1]": "[input]"})
        param.process_model(model)
        param.process_geometry(geometry)
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 10, "r_p": 10}
        spatial_methods = model.default_spatial_methods
        solver = model.default_solver
        sim = pybamm.Simulation(
            model=model,
            geometry=geometry,
            parameter_values=param,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
        )
        inputs = {"Negative electrode conductivity [S.m-1]": 0.1}
        sim.solve(t_eval=np.linspace(0, 10, 10), inputs=inputs)
        time = sim.solution["Time [h]"](sim.solution.t)
        self.assertEqual(len(time), 10)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
