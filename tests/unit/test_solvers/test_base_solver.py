#
# Tests for the Base Solver class
#
import pybamm

import unittest


class TestBaseSolver(unittest.TestCase):
    def test_base_solver_init(self):
        solver = pybamm.BaseSolver(tol=1e-4)
        self.assertEqual(solver.tol, 1e-4)

        solver.tol = 1e-5
        self.assertEqual(solver.tol, 1e-5)

        model = pybamm.BaseModel()
        with self.assertRaises(NotImplementedError):
            solver.solve(model, None)
            solver.step(model, None)
            solver.compute_solution(None, None)
            solver.set_up(None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
