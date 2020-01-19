#
# Tests for the Base Solver class
#
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
            solver.step(model, None)
        with self.assertRaisesRegex(pybamm.ModelError, "Cannot solve empty model"):
            solver.solve(model, None)

    def test_set_external_variables(self):
        options = {"thermal": "x-full", "external submodels": ["thermal"]}
        model = pybamm.lithium_ion.SPM(options)
        sim = pybamm.Simulation(model)
        sim.build()
        solver = pybamm.BaseSolver()

        T = np.ones((60, 1))
        external_variables = {"Cell temperature": T}
        solver.set_external_variables(sim.built_model, external_variables)

    def test_find_consistent_initial_conditions(self):
        # Simple system: a single algebraic equation
        def rhs(t, y):
            return np.array([])

        def algebraic(t, y):
            return y + 2

        solver = pybamm.BaseSolver()
        y0 = np.array([2])
        init_cond = solver.calculate_consistent_initial_conditions(rhs, algebraic, y0)
        np.testing.assert_array_equal(init_cond, -2)

        # More complicated system
        vec = np.array([0.0, 1.0, 1.5, 2.0])

        def rhs(t, y):
            return y[0:1]

        def algebraic(t, y):
            return (y[1:] - vec[1:]) ** 2

        y0 = np.zeros_like(vec)
        init_cond = solver.calculate_consistent_initial_conditions(rhs, algebraic, y0)
        np.testing.assert_array_almost_equal(init_cond, vec)

        # With jacobian
        def jac_dense(t, y):
            return 2 * np.hstack([np.zeros((3, 1)), np.diag(y[1:] - vec[1:])])

        init_cond = solver.calculate_consistent_initial_conditions(
            rhs, algebraic, y0, jac_dense
        )
        np.testing.assert_array_almost_equal(init_cond, vec)

        # With sparse jacobian
        def jac_sparse(t, y):
            return 2 * csr_matrix(
                np.hstack([np.zeros((3, 1)), np.diag(y[1:] - vec[1:])])
            )

        init_cond = solver.calculate_consistent_initial_conditions(
            rhs, algebraic, y0, jac_sparse
        )
        np.testing.assert_array_almost_equal(init_cond, vec)

    def test_fail_consistent_initial_conditions(self):
        def rhs(t, y):
            return np.array([])

        def algebraic(t, y):
            # algebraic equation has no root
            return y ** 2 + 1

        solver = pybamm.BaseSolver(root_method="hybr")
        y0 = np.array([2])

        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Could not find consistent initial conditions: The iteration is not making",
        ):
            solver.calculate_consistent_initial_conditions(rhs, algebraic, y0)
        solver = pybamm.BaseSolver()
        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Could not find consistent initial conditions: solver terminated",
        ):
            solver.calculate_consistent_initial_conditions(rhs, algebraic, y0)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
