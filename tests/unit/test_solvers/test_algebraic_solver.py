#
# Tests for the Algebraic Solver class
#

import pybamm
import unittest
import numpy as np
from tests import get_discretisation_for_testing


class TestAlgebraicSolver(unittest.TestCase):
    def test_algebraic_solver_init(self):
        solver = pybamm.AlgebraicSolver(
            method="hybr", tol=1e-4, extra_options={"maxfev": 100}
        )
        self.assertEqual(solver.method, "hybr")
        self.assertEqual(solver.extra_options, {"maxfev": 100})
        self.assertEqual(solver.tol, 1e-4)

        solver.method = "krylov"
        self.assertEqual(solver.method, "krylov")
        solver.tol = 1e-5
        self.assertEqual(solver.tol, 1e-5)

    def test_wrong_solver(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: var}
        model.algebraic = {var: var - 1}

        # test errors
        solver = pybamm.AlgebraicSolver()
        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Cannot use algebraic solver to solve model with time derivatives",
        ):
            solver.solve(model)

    def test_simple_root_find(self):
        # Simple system: a single algebraic equation
        class Model(pybamm.BaseModel):
            y0 = np.array([2])
            rhs = {}
            jac_algebraic_eval = None
            len_rhs_and_alg = 1

            def __init__(self):
                super().__init__()
                self.convert_to_format = "python"

            def algebraic_eval(self, t, y, inputs):
                return y + 2

        # Try passing extra options to solver
        solver = pybamm.AlgebraicSolver(extra_options={"maxiter": 100})
        model = Model()
        solution = solver._integrate(model, np.array([0]))
        np.testing.assert_array_equal(solution.y, -2)

        # Relax options and see worse results
        solver = pybamm.AlgebraicSolver(extra_options={"ftol": 1})
        solution = solver._integrate(model, np.array([0]))
        self.assertNotEqual(solution.y, -2)

    def test_root_find_fail(self):
        class Model(pybamm.BaseModel):
            y0 = np.array([2])
            rhs = {}
            jac_algebraic_eval = None
            len_rhs_and_alg = 1

            def __init__(self):
                super().__init__()
                self.convert_to_format = "python"

            def algebraic_eval(self, t, y, inputs):
                # algebraic equation has no real root
                return y**2 + 1

        model = Model()

        solver = pybamm.AlgebraicSolver(method="hybr")
        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Could not find acceptable solution: The iteration is not making",
        ):
            solver._integrate(model, np.array([0]))

        solver = pybamm.AlgebraicSolver()
        with self.assertRaisesRegex(
            pybamm.SolverError, "Could not find acceptable solution: solver terminated"
        ):
            solver._integrate(model, np.array([0]))

    def test_with_jacobian(self):
        A = np.array([[4, 3], [1, -1]])
        b = np.array([0, 7])

        class Model(pybamm.BaseModel):
            y0 = np.zeros(2)
            rhs = {}
            len_rhs_and_alg = 2

            def __init__(self):
                super().__init__()
                self.convert_to_format = "python"

            def algebraic_eval(self, t, y, inputs):
                return A @ y - b

            def jac_algebraic_eval(self, t, y, inputs):
                return A

        model = Model()
        sol = np.array([3, -4])[:, np.newaxis]

        solver = pybamm.AlgebraicSolver()
        solution = solver._integrate(model, np.array([0]))
        np.testing.assert_array_almost_equal(solution.y, sol)

    def test_model_solver(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.algebraic = {var1: var1 - 3, var2: 2 * var1 - var2}
        model.initial_conditions = {var1: pybamm.Scalar(1), var2: pybamm.Scalar(4)}
        model.variables = {"var1": var1, "var2": var2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        sol = np.concatenate((np.ones(100) * 3, np.ones(100) * 6))[:, np.newaxis]

        # Solve
        solver = pybamm.AlgebraicSolver()
        solution = solver.solve(model)
        np.testing.assert_array_equal(
            model.variables["var1"].evaluate(t=None, y=solution.y), sol[:100]
        )
        np.testing.assert_array_equal(
            model.variables["var2"].evaluate(t=None, y=solution.y), sol[100:]
        )

        # Test without Jacobian
        model.use_jacobian = False
        solver.models_set_up = set()
        solution_no_jac = solver.solve(model)
        np.testing.assert_array_equal(
            model.variables["var1"].evaluate(t=None, y=solution_no_jac.y), sol[:100]
        )
        np.testing.assert_array_equal(
            model.variables["var2"].evaluate(t=None, y=solution_no_jac.y), sol[100:]
        )

    def test_model_solver_least_squares(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.algebraic = {var1: var1 - 3, var2: 2 * var1 - var2}
        model.initial_conditions = {var1: pybamm.Scalar(1), var2: pybamm.Scalar(4)}
        model.variables = {"var1": var1, "var2": var2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        sol = np.concatenate((np.ones(100) * 3, np.ones(100) * 6))[:, np.newaxis]

        # Solve
        solver = pybamm.AlgebraicSolver("lsq")
        solution = solver.solve(model)
        np.testing.assert_array_almost_equal(
            model.variables["var1"].evaluate(t=None, y=solution.y), sol[:100]
        )
        np.testing.assert_array_almost_equal(
            model.variables["var2"].evaluate(t=None, y=solution.y), sol[100:]
        )

        # Test without jacobian and with a different method
        model.use_jacobian = False
        solver = pybamm.AlgebraicSolver("lsq__trf")
        solution_no_jac = solver.solve(model)
        np.testing.assert_array_almost_equal(
            model.variables["var1"].evaluate(t=None, y=solution_no_jac.y), sol[:100]
        )
        np.testing.assert_array_almost_equal(
            model.variables["var2"].evaluate(t=None, y=solution_no_jac.y), sol[100:]
        )

    def test_model_solver_minimize(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.algebraic = {var1: var1 - 3, var2: 2 * var1 - var2}
        model.initial_conditions = {var1: pybamm.Scalar(1), var2: pybamm.Scalar(4)}
        model.variables = {"var1": var1, "var2": var2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        sol = np.concatenate((np.ones(100) * 3, np.ones(100) * 6))[:, np.newaxis]

        # Solve
        solver = pybamm.AlgebraicSolver("minimize", tol=1e-8)
        solution = solver.solve(model)
        np.testing.assert_array_almost_equal(
            model.variables["var1"].evaluate(t=None, y=solution.y), sol[:100]
        )
        np.testing.assert_array_almost_equal(
            model.variables["var2"].evaluate(t=None, y=solution.y), sol[100:]
        )

        # Test without jacobian and with a different method
        model.use_jacobian = False
        solver = pybamm.AlgebraicSolver("minimize__BFGS")
        solution_no_jac = solver.solve(model)
        np.testing.assert_array_almost_equal(
            model.variables["var1"].evaluate(t=None, y=solution_no_jac.y), sol[:100]
        )
        np.testing.assert_array_almost_equal(
            model.variables["var2"].evaluate(t=None, y=solution_no_jac.y), sol[100:]
        )

    def test_model_solver_least_squares_with_bounds(self):
        # Note: we need a better test case to test this functionality properly
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1", bounds=(0, 10))
        model.algebraic = {var1: pybamm.sin(var1) + 1}
        model.initial_conditions = {var1: pybamm.Scalar(3)}
        model.variables = {"var1": var1}

        # Solve
        solver = pybamm.AlgebraicSolver("lsq", tol=1e-5)
        solution = solver.solve(model)
        np.testing.assert_array_almost_equal(
            model.variables["var1"].evaluate(t=None, y=solution.y),
            3 * np.pi / 2,
            decimal=2,
        )

    def test_model_solver_minimize_with_bounds(self):
        # Note: we need a better test case to test this functionality properly
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1", bounds=(0, 10))
        model.algebraic = {var1: pybamm.sin(var1) + 1}
        model.initial_conditions = {var1: pybamm.Scalar(3)}
        model.variables = {"var1": var1}

        # Solve
        solver = pybamm.AlgebraicSolver("minimize", tol=1e-16)
        solution = solver.solve(model)
        np.testing.assert_array_almost_equal(
            model.variables["var1"].evaluate(t=None, y=solution.y),
            3 * np.pi / 2,
            decimal=4,
        )

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
        solver = pybamm.AlgebraicSolver()
        solution = solver.solve(model, t_eval)

        sol = np.vstack((3 * t_eval, 6 * t_eval))
        np.testing.assert_array_equal(solution.y, sol)
        np.testing.assert_array_equal(
            model.variables["var1"].evaluate(t=t_eval, y=solution.y).flatten(),
            sol[0, :],
        )
        np.testing.assert_array_equal(
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
        solver = pybamm.AlgebraicSolver()
        solution = solver.solve(model, np.linspace(0, 1, 10), inputs={"value": 7})
        np.testing.assert_array_equal(solution.y, -7)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
