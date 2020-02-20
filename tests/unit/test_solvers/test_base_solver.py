#
# Tests for the Base Solver class
#
import casadi
import pybamm
import numpy as np
from scipy.sparse import csr_matrix

import unittest


class TestBaseSolver(unittest.TestCase):
    def test_base_solver_init(self):
        solver = pybamm.BaseSolver(rtol=1e-2, atol=1e-4)
        self.assertEqual(solver.rtol, 1e-2)
        self.assertEqual(solver.atol, 1e-4)

        solver.rtol = 1e-5
        self.assertEqual(solver.rtol, 1e-5)
        solver.rtol = 1e-7
        self.assertEqual(solver.rtol, 1e-7)

    def test_step_or_solve_empty_model(self):
        model = pybamm.BaseModel()
        solver = pybamm.BaseSolver()
        with self.assertRaisesRegex(pybamm.ModelError, "Cannot step empty model"):
            solver.step(None, model, None)
        with self.assertRaisesRegex(pybamm.ModelError, "Cannot solve empty model"):
            solver.solve(model, None)

    def test_nonmonotonic_teval(self):
        solver = pybamm.BaseSolver(rtol=1e-2, atol=1e-4)
        model = pybamm.BaseModel()
        a = pybamm.Scalar(0)
        model.rhs = {a: a}
        with self.assertRaisesRegex(
            pybamm.SolverError, "t_eval must increase monotonically"
        ):
            solver.solve(model, np.array([1, 2, 3, 2]))

    def test_ode_solver_fail_with_dae(self):
        model = pybamm.BaseModel()
        a = pybamm.Scalar(1)
        model.algebraic = {a: a}
        model.concatenated_initial_conditions = pybamm.Scalar(0)
        solver = pybamm.ScipySolver()
        with self.assertRaisesRegex(pybamm.SolverError, "Cannot use ODE solver"):
            solver.set_up(model)

    def test_find_consistent_initial_conditions(self):
        # Simple system: a single algebraic equation
        class ScalarModel:
            def __init__(self):
                self.concatenated_initial_conditions = np.array([[2]])
                self.jac_algebraic_eval = None
                self.timescale = 1
                t = casadi.MX.sym("t")
                y = casadi.MX.sym("y")
                u = casadi.MX.sym("u")
                self.casadi_algebraic = casadi.Function(
                    "alg", [t, y, u], [self.algebraic_eval(t, y)]
                )

            def rhs_eval(self, t, y):
                return np.array([])

            def algebraic_eval(self, t, y):
                return y + 2

        solver = pybamm.BaseSolver(root_method="lm")
        model = ScalarModel()
        init_cond = solver.calculate_consistent_state(model)
        np.testing.assert_array_equal(init_cond, -2)
        # with casadi
        solver_with_casadi = pybamm.BaseSolver(root_method="casadi")
        model = ScalarModel()
        init_cond = solver_with_casadi.calculate_consistent_state(model)
        np.testing.assert_array_equal(init_cond, -2)

        # More complicated system
        vec = np.array([0.0, 1.0, 1.5, 2.0])

        class VectorModel:
            def __init__(self):
                self.concatenated_initial_conditions = np.zeros_like(vec)
                self.jac_algebraic_eval = None
                self.timescale = 1
                t = casadi.MX.sym("t")
                y = casadi.MX.sym("y", vec.size)
                u = casadi.MX.sym("u")
                self.casadi_algebraic = casadi.Function(
                    "alg", [t, y, u], [self.algebraic_eval(t, y)]
                )

            def rhs_eval(self, t, y):
                return y[0:1]

            def algebraic_eval(self, t, y):
                return (y[1:] - vec[1:]) ** 2

        model = VectorModel()
        init_cond = solver.calculate_consistent_state(model)
        np.testing.assert_array_almost_equal(init_cond, vec)
        # with casadi
        init_cond = solver_with_casadi.calculate_consistent_state(model)
        np.testing.assert_array_almost_equal(init_cond, vec)

        # With jacobian
        def jac_dense(t, y):
            return 2 * np.hstack([np.zeros((3, 1)), np.diag(y[1:] - vec[1:])])

        model.jac_algebraic_eval = jac_dense
        init_cond = solver.calculate_consistent_state(model)
        np.testing.assert_array_almost_equal(init_cond, vec)

        # With sparse jacobian
        def jac_sparse(t, y):
            return 2 * csr_matrix(
                np.hstack([np.zeros((3, 1)), np.diag(y[1:] - vec[1:])])
            )

        model.jac_algebraic_eval = jac_sparse
        init_cond = solver.calculate_consistent_state(model)
        np.testing.assert_array_almost_equal(init_cond, vec)

    def test_fail_consistent_initial_conditions(self):
        class Model:
            def __init__(self):
                self.concatenated_initial_conditions = np.array([2])
                self.jac_algebraic_eval = None
                self.timescale = 1
                t = casadi.MX.sym("t")
                y = casadi.MX.sym("y")
                u = casadi.MX.sym("u")
                self.casadi_algebraic = casadi.Function(
                    "alg", [t, y, u], [self.algebraic_eval(t, y)]
                )

            def rhs_eval(self, t, y):
                return np.array([])

            def algebraic_eval(self, t, y):
                # algebraic equation has no root
                return y ** 2 + 1

        solver = pybamm.BaseSolver(root_method="hybr")

        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Could not find consistent initial conditions: The iteration is not making",
        ):
            solver.calculate_consistent_state(Model())
        solver = pybamm.BaseSolver(root_method="lm")
        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Could not find consistent initial conditions: solver terminated",
        ):
            solver.calculate_consistent_state(Model())
        # with casadi
        solver = pybamm.BaseSolver(root_method="casadi")
        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Could not find consistent initial conditions: solver terminated",
        ):
            solver.calculate_consistent_state(Model())

    def test_time_too_short(self):
        solver = pybamm.BaseSolver()
        model = pybamm.BaseModel()
        v = pybamm.StateVector(slice(0, 1))
        model.rhs = {v: v}
        with self.assertRaisesRegex(
            pybamm.SolverError, "It looks like t_eval might be dimensionless"
        ):
            solver.solve(model, np.linspace(0, 0.1))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
