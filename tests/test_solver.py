from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestSolver(unittest.TestCase):
    def test_solver_basic(self):
        with self.assertRaises(NotImplementedError):
            pybamm.Solver(integrator="not an integrator")
        with self.assertRaises(NotImplementedError):
            pybamm.Solver(
                spatial_discretisation="not a spatial discretisation"
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
