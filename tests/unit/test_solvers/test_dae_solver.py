#
# Tests for the DAE Solver class
#
import pybamm
import unittest
import numpy as np


class TestDaeSolver(unittest.TestCase):
    def test_find_consistent_initial_conditions(self):
        # Simple system: a single algebraic equation
        def rhs(t, y):
            return np.array([])

        def algebraic(t, y):
            return y + 2

        solver = pybamm.DaeSolver()
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

    def test_fail_consistent_initial_conditions(self):
        def rhs(t, y):
            return np.array([])

        def algebraic(t, y):
            # algebraic equation has no root
            return y ** 2 + 1

        solver = pybamm.DaeSolver()
        y0 = np.array([2])
        with self.assertRaises(pybamm.SolverError):
            solver.calculate_consistent_initial_conditions(rhs, algebraic, y0)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
