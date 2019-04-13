#
# Tests for the Base Solver class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestBaseSolver(unittest.TestCase):
    def test_base_solver_init(self):
        solver = pybamm.BaseSolver(tol=1e-4)
        self.assertEqual(solver.tol, 1e-4)
        self.assertEqual(solver.t, None)
        self.assertEqual(solver.y, None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
