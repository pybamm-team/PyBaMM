#
# Tests for the Casadi Algebraic Solver class
#
import casadi
import pybamm
import unittest
import numpy as np
from scipy.optimize import least_squares
import tests


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

    def test_simple_root_find_correct_initial_guess(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + 2}
        # initial guess gives right answer
        model.initial_conditions = {var: -2}

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
            p = casadi.MX.sym("p")
            rhs = {}
            casadi_algebraic = casadi.Function("alg", [t, y, p], [y**2 + 1])
            bounds = (np.array([-np.inf]), np.array([np.inf]))
            interpolant_extrapolation_events_eval = []

            def algebraic_eval(self, t, y, inputs):
                # algebraic equation has no real root
                return y**2 + 1

        model = Model()

        solver = pybamm.CasadiAlgebraicSolver()
        with self.assertRaisesRegex(
            pybamm.SolverError, "Could not find acceptable solution: .../casadi"
        ):
            solver._integrate(model, np.array([0]), {})
        solver = pybamm.CasadiAlgebraicSolver(extra_options={"error_on_fail": False})
        with self.assertRaisesRegex(
            pybamm.SolverError, "Could not find acceptable solution: solver terminated"
        ):
            solver._integrate(model, np.array([0]), {})

        # Model returns Nan
        class NaNModel:
            y0 = np.array([-2])
            t = casadi.MX.sym("t")
            y = casadi.MX.sym("y")
            p = casadi.MX.sym("p")
            rhs = {}
            casadi_algebraic = casadi.Function("alg", [t, y, p], [y**0.5])
            bounds = (np.array([-np.inf]), np.array([np.inf]))
            interpolant_extrapolation_events_eval = []

            def algebraic_eval(self, t, y, inputs):
                # algebraic equation has no real root
                return y**0.5

        model = NaNModel()
        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Could not find acceptable solution: solver returned NaNs",
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
        np.testing.assert_array_almost_equal(solution["var1"].data.flatten(), sol[0, :])
        np.testing.assert_array_almost_equal(solution["var2"].data.flatten(), sol[1, :])

    def test_model_solver_with_time_not_changing(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.algebraic = {var1: var1 - 3, var2: 2 * var1 - var2}
        model.initial_conditions = {var1: pybamm.Scalar(1), var2: pybamm.Scalar(4)}
        model.variables = {"var1": var1, "var2": var2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        t_eval = np.linspace(0, 1)
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(model, t_eval)

        sol = np.vstack((3 + 0 * t_eval, 6 + 0 * t_eval))
        np.testing.assert_array_almost_equal(solution.y, sol)

    def test_model_solver_with_bounds(self):
        # Note: we need a better test case to test this functionality properly
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1", bounds=(0, 10))
        model.algebraic = {var1: pybamm.sin(var1) + 1}
        model.initial_conditions = {var1: pybamm.Scalar(3)}
        model.variables = {"var1": var1}

        # Solve
        solver = pybamm.CasadiAlgebraicSolver(tol=1e-12)
        solution = solver.solve(model)
        np.testing.assert_array_almost_equal(solution["var1"].data, 3 * np.pi / 2)

    def test_solve_with_input(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + pybamm.InputParameter("param")}
        model.initial_conditions = {var: 2}

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(model, np.linspace(0, 1, 10), inputs={"param": 7})
        np.testing.assert_array_equal(solution.y, -7)


class TestCasadiAlgebraicSolverSensitivity(unittest.TestCase):
    def test_solve_with_symbolic_input(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + pybamm.InputParameter("param")}
        model.initial_conditions = {var: 2}
        model.variables = {"var": var}

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(
            model, [0], inputs={"param": 7}, calculate_sensitivities=True
        )
        np.testing.assert_array_equal(solution["var"].data, -7)

        solution = solver.solve(
            model, [0], inputs={"param": 3}, calculate_sensitivities=True
        )
        np.testing.assert_array_equal(solution["var"].data, -3)
        np.testing.assert_array_equal(solution["var"].sensitivities["param"], -1)

    def test_least_squares_fit(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var", domain="negative electrode")
        model = pybamm.BaseModel()
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        model.algebraic = {var: (var - p)}
        model.initial_conditions = {var: 3}
        model.variables = {"objective": (var - q) ** 2 + (p - 3) ** 2}

        # create discretisation
        disc = tests.get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiAlgebraicSolver()

        def objective(x):
            solution = solver.solve(
                model, [0], inputs={"p": x[0], "q": x[1]}, calculate_sensitivities=True
            )
            return solution["objective"].data.flatten()

        # without Jacobian
        lsq_sol = least_squares(objective, [2, 2], method="lm")
        np.testing.assert_array_almost_equal(lsq_sol.x, [3, 3], decimal=3)

        def jac(x):
            solution = solver.solve(
                model, [0], inputs={"p": x[0], "q": x[1]}, calculate_sensitivities=True
            )
            return np.concatenate(
                [solution["objective"].sensitivities[name] for name in ["p", "q"]],
                axis=1,
            )

        # with Jacobian
        lsq_sol = least_squares(objective, [2, 2], jac=jac, method="lm")
        np.testing.assert_array_almost_equal(lsq_sol.x, [3, 3], decimal=3)

    def test_solve_with_symbolic_input_1D_scalar_input(self):
        var = pybamm.Variable("var", "negative electrode")
        model = pybamm.BaseModel()
        param = pybamm.InputParameter("param")
        model.algebraic = {var: var + param}
        model.initial_conditions = {var: 2}
        model.variables = {"var": var}

        # create discretisation
        disc = tests.get_discretisation_for_testing()
        disc.process_model(model)

        # Solve - scalar input
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(
            model, [0], inputs={"param": 7}, calculate_sensitivities=True
        )
        np.testing.assert_array_equal(solution["var"].data, -7)

        solution = solver.solve(
            model, [0], inputs={"param": 3}, calculate_sensitivities=True
        )
        np.testing.assert_array_equal(solution["var"].data, -3)
        np.testing.assert_array_equal(solution["var"].sensitivities["param"], -1)

    def test_solve_with_symbolic_input_1D_vector_input(self):
        var = pybamm.Variable("var", "negative electrode")
        model = pybamm.BaseModel()
        param = pybamm.InputParameter("param", "negative electrode")
        model.algebraic = {var: var + param}
        model.initial_conditions = {var: 2}
        model.variables = {"var": var}

        # create discretisation
        disc = tests.get_discretisation_for_testing()
        disc.process_model(model)

        # Solve - scalar input
        solver = pybamm.CasadiAlgebraicSolver()
        n = disc.mesh["negative electrode"].npts

        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(
            model, [0], inputs={"param": 3 * np.ones(n)}, calculate_sensitivities=True
        )
        np.testing.assert_array_almost_equal(
            solution["var"].sensitivities["param"], -np.eye(40)
        )
        np.testing.assert_array_almost_equal(solution["var"].data, -3)
        p = np.linspace(0, 1, n)[:, np.newaxis]
        solution = solver.solve(
            model, [0], inputs={"param": p}, calculate_sensitivities=True
        )
        np.testing.assert_array_almost_equal(solution["var"].data, -p)
        np.testing.assert_array_almost_equal(
            solution["var"].sensitivities["param"], -np.eye(40)
        )

    def test_solve_with_symbolic_input_in_initial_conditions(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + 2}
        model.initial_conditions = {var: pybamm.InputParameter("param")}
        model.variables = {"var": var}

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(
            model, [0], inputs={"param": 7}, calculate_sensitivities=True
        )
        np.testing.assert_array_equal(solution["var"].data, -2)
        np.testing.assert_array_equal(solution["var"].sensitivities["param"], 0)
        solution = solver.solve(
            model, [0], inputs={"param": 3}, calculate_sensitivities=True
        )
        np.testing.assert_array_equal(solution["var"].data, -2)

    def test_least_squares_fit_input_in_initial_conditions(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var", domain="negative electrode")
        model = pybamm.BaseModel()
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        model.algebraic = {var: (var - p)}
        model.initial_conditions = {var: p}
        model.variables = {"objective": (var - q) ** 2 + (p - 3) ** 2}

        # create discretisation
        disc = tests.get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiAlgebraicSolver()

        def objective(x):
            solution = solver.solve(
                model, [0], inputs={"p": x[0], "q": x[1]}, calculate_sensitivities=True
            )
            return solution["objective"].data.flatten()

        # without Jacobian
        lsq_sol = least_squares(objective, [2, 2], method="lm")
        np.testing.assert_array_almost_equal(lsq_sol.x, [3, 3], decimal=3)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
