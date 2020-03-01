#
# Tests for the Casadi Algebraic Solver class
#
import casadi
import pybamm
import unittest
import numpy as np


class TestCasadiAlgebraicSolver(unittest.TestCase):
    def test_algebraic_solver_init(self):
        solver = pybamm.CasadiAlgebraicSolver(tol=1e-4)
        self.assertEqual(solver.tol, 1e-4)

        solver.tol = 1e-5
        self.assertEqual(solver.tol, 1e-5)

    def test_simple_root_find(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + 2}
        model.initial_conditions = {var: 2}

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(model, np.linspace(0, 1, 10))
        np.testing.assert_array_equal(solution.y, -2)

    def test_root_find_fail(self):
        class Model:
            y0 = np.array([2])
            t = casadi.MX.sym("t")
            y = casadi.MX.sym("y")
            u = casadi.MX.sym("u")
            casadi_algebraic = casadi.Function("alg", [t, y, u], [y ** 2 + 1])

            def algebraic_eval(self, t, y):
                # algebraic equation has no real root
                return y ** 2 + 1

        model = Model()

        solver = pybamm.CasadiAlgebraicSolver()
        with self.assertRaisesRegex(
            pybamm.SolverError, "Could not find acceptable solution: .../casadi",
        ):
            solver._integrate(model, np.array([0]), {})

    def test_model_solver_with_time(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.algebraic = {var1: var1 - 3 * pybamm.t, var2: 2 * var1 - var2}
        model.initial_conditions = {var1: pybamm.Scalar(1), var2: pybamm.Scalar(4)}
        model.variables = {"var1": var1, "var2": var2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        t_eval = np.linspace(0, 1)
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(model, t_eval)

        sol = np.vstack((3 * t_eval, 6 * t_eval))
        np.testing.assert_array_almost_equal(solution.y, sol)
        np.testing.assert_array_almost_equal(
            model.variables["var1"].evaluate(t=t_eval, y=solution.y).flatten(),
            sol[0, :],
        )
        np.testing.assert_array_almost_equal(
            model.variables["var2"].evaluate(t=t_eval, y=solution.y).flatten(),
            sol[1, :],
        )

    def test_solve_with_input(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + pybamm.InputParameter("value")}
        model.initial_conditions = {var: 2}

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(model, np.linspace(0, 1, 10), inputs={"value": 7})
        np.testing.assert_array_equal(solution.y, -7)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
