#
# Tests for the Scipy Solver class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest
import numpy as np
import tests.shared as shared


class TestScipySolver(unittest.TestCase):
    def test_integrate(self):
        # Constant
        solver = pybamm.ScipySolver(tol=1e-8, method="RK45")

        def constant_growth(t, y):
            return 0.5 * np.ones_like(y)

        y0 = np.array([0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(constant_growth, y0, t_eval)
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])

        # Exponential decay
        solver = pybamm.ScipySolver(tol=1e-8, method="BDF")

        def exponential_decay(t, y):
            return -0.1 * y

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(exponential_decay, y0, t_eval)
        np.testing.assert_allclose(y_sol[0], np.exp(-0.1 * t_sol))

    def test_model_solver(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: pybamm.Scalar(0.1) * var}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        defaults = shared.TestDefaults1DMacro()
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume,
            "separator": pybamm.FiniteVolume,
            "positive electrode": pybamm.FiniteVolume,
        }
        disc = pybamm.Discretisation(defaults.mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.ScipySolver(tol=1e-8, method="RK45")
        t_eval = np.linspace(0, 1, 100)
        solver.solve(model, t_eval)
        np.testing.assert_array_equal(solver.t, t_eval)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
