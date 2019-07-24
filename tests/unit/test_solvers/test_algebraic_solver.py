#
# Tests for the Algebraic Solver class
#
import pybamm
import unittest
import numpy as np


class TestAlgebraicSolver(unittest.TestCase):
    def test_algebraic_solver_init(self):
        solver = pybamm.AlgebraicSolver(method="hybr", tol=1e-4)
        self.assertEqual(solver.method, "hybr")
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
        def algebraic(y):
            return y + 2

        solver = pybamm.AlgebraicSolver()
        y0 = np.array([2])
        solution = solver.root(algebraic, y0)
        np.testing.assert_array_equal(solution.y, -2)

    def test_root_find_fail(self):
        def algebraic(y):
            # algebraic equation has no real root
            return y ** 2 + 1

        solver = pybamm.AlgebraicSolver(method="hybr")
        y0 = np.array([2])

        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Could not find acceptable solution: The iteration is not making",
        ):
            solver.root(algebraic, y0)
        solver = pybamm.AlgebraicSolver()
        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Could not find acceptable solution: solver terminated",
        ):
            solver.root(algebraic, y0)

    def test_with_jacobian(self):
        A = np.array([[4, 3], [1, -1]])
        b = np.array([0, 7])

        def algebraic(y):
            return A @ y - b

        def jac(t, y):
            return A

        y0 = np.zeros(2)
        sol = np.array([3, -4])

        solver = pybamm.AlgebraicSolver()

        solution_no_jac = solver.root(algebraic, y0)
        solution_with_jac = solver.root(algebraic, y0, jacobian=jac)

        np.testing.assert_array_almost_equal(solution_no_jac.y, sol)
        np.testing.assert_array_almost_equal(solution_with_jac.y, sol)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
