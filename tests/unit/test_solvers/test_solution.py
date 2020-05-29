#
# Tests for the Solution class
#
import pybamm
import unittest
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tests import get_discretisation_for_testing


class TestSolution(unittest.TestCase):
    def test_init(self):
        t = np.linspace(0, 1)
        y = np.tile(t, (20, 1))
        sol = pybamm.Solution(t, y)
        np.testing.assert_array_equal(sol.t, t)
        np.testing.assert_array_equal(sol.y, y)
        self.assertEqual(sol.t_event, None)
        self.assertEqual(sol.y_event, None)
        self.assertEqual(sol.termination, "final time")
        self.assertEqual(sol.inputs, {})
        self.assertIsInstance(sol.model, pybamm.BaseModel)

        with self.assertRaisesRegex(AttributeError, "sub solutions"):
            print(sol.sub_solutions)

    def test_append(self):
        # Set up first solution
        t1 = np.linspace(0, 1)
        y1 = np.tile(t1, (20, 1))
        sol1 = pybamm.Solution(t1, y1)
        sol1.solve_time = 1.5
        sol1.model = pybamm.BaseModel()
        sol1.inputs = {"a": 1}

        # Set up second solution
        t2 = np.linspace(1, 2)
        y2 = np.tile(t2, (20, 1))
        sol2 = pybamm.Solution(t2, y2)
        sol2.solve_time = 1
        sol2.inputs = {"a": 2}
        sol1.append(sol2, create_sub_solutions=True)

        # Test
        self.assertEqual(sol1.solve_time, 2.5)
        np.testing.assert_array_equal(sol1.t, np.concatenate([t1, t2[1:]]))
        np.testing.assert_array_equal(sol1.y, np.concatenate([y1, y2[:, 1:]], axis=1))
        np.testing.assert_array_equal(
            sol1.inputs["a"],
            np.concatenate([1 * np.ones_like(t1), 2 * np.ones_like(t2[1:])])[
                np.newaxis, :
            ],
        )

        # Test sub-solutions
        self.assertEqual(len(sol1.sub_solutions), 2)
        np.testing.assert_array_equal(sol1.sub_solutions[0].t, t1)
        np.testing.assert_array_equal(sol1.sub_solutions[1].t, t2)
        self.assertEqual(sol1.sub_solutions[0].model, sol1.model)
        np.testing.assert_array_equal(
            sol1.sub_solutions[0].inputs["a"], 1 * np.ones_like(t1)[np.newaxis, :]
        )
        self.assertEqual(sol1.sub_solutions[1].model, sol2.model)
        np.testing.assert_array_equal(
            sol1.sub_solutions[1].inputs["a"], 2 * np.ones_like(t2)[np.newaxis, :]
        )

    def test_total_time(self):
        sol = pybamm.Solution([], None)
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

        disc = pybamm.Discretisation()
        disc.process_model(model)
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

    def test_save(self):
        model = pybamm.BaseModel()
        # create both 1D and 2D variables
        c = pybamm.Variable("c")
        d = pybamm.Variable("d", domain="negative electrode")
        model.rhs = {c: -c, d: 1}
        model.initial_conditions = {c: 1, d: 2}
        model.variables = {"c": c, "d": d, "2c": 2 * c}

        disc = get_discretisation_for_testing()
        disc.process_model(model)
        solution = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

        # test save data
        with self.assertRaises(ValueError):
            solution.save_data("test.pickle")
        # set variables first then save
        solution.update(["c", "d"])
        solution.save_data("test.pickle")
        data_load = pybamm.load("test.pickle")
        np.testing.assert_array_equal(solution.data["c"], data_load["c"])
        np.testing.assert_array_equal(solution.data["d"], data_load["d"])

        # to matlab
        solution.save_data("test.mat", to_format="matlab")
        data_load = loadmat("test.mat")
        np.testing.assert_array_equal(solution.data["c"], data_load["c"].flatten())
        np.testing.assert_array_equal(solution.data["d"], data_load["d"])

        # to csv
        with self.assertRaisesRegex(
            ValueError, "only 0D variables can be saved to csv"
        ):
            solution.save_data("test.csv", to_format="csv")
        # only save "c" and "2c"
        solution.save_data("test.csv", ["c", "2c"], to_format="csv")
        # read csv
        df = pd.read_csv("test.csv")
        np.testing.assert_array_almost_equal(df["c"], solution.data["c"])
        np.testing.assert_array_almost_equal(df["2c"], solution.data["2c"])

        # test save whole solution
        solution.save("test.pickle")
        solution_load = pybamm.load("test.pickle")
        self.assertEqual(solution.model.name, solution_load.model.name)
        np.testing.assert_array_equal(solution["c"].entries, solution_load["c"].entries)
        np.testing.assert_array_equal(solution["d"].entries, solution_load["d"].entries)

    def test_solution_evals_with_inputs(self):
        model = pybamm.lithium_ion.SPM()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.update({"Negative electrode conductivity [S.m-1]": "[input]"})
        param.process_model(model)
        param.process_geometry(geometry)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 10, var.r_p: 10}
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
