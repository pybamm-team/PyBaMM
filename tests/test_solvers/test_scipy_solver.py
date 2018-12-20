#
# Tests for the Scipy Solver class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest
import numpy as np


class TestScipySolver(unittest.TestCase):
    def test_integrate(self):
        solver = pybamm.ScipySolver(tol=1e-8)

        # Constant
        def constant_derivs(t, y):
            return np.ones_like(y)

        y0 = np.array([0])
        t_eval = np.linspace(0, 1, 100)
        sol = solver.integrate(constant_derivs, y0, t_eval)
        np.testing.assert_allclose(sol.t, t_eval)
        np.testing.assert_allclose(sol.t, sol.y[0])

        # Exponential decay
        def exponential_derivs(t, y):
            return -y

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        sol = solver.integrate(exponential_derivs, y0, t_eval)
        np.testing.assert_allclose(sol.y[0], np.exp(-sol.t))

    @unittest.skip("Base model not yet implemented")
    def test_model_solver(self):
        pass


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
