#
# Tests for the Scipy Solver class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest
import numpy as np
from tests import get_mesh_for_testing


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

    def test_integrate_with_event(self):
        # Constant
        solver = pybamm.ScipySolver(tol=1e-8, method="RK45")

        def constant_growth(t, y):
            return 0.5 * np.ones_like(y)

        def y_eq_2_ysq_eq_5(t, y):
            return y - 2

        y0 = np.array([0])
        t_eval = np.linspace(0, 10, 100)
        t_sol, y_sol = solver.integrate(
            constant_growth, y0, t_eval, events=y_eq_2_ysq_eq_5
        )
        self.assertLess(len(t_sol), len(t_eval))
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])

        # Exponential decay
        solver = pybamm.ScipySolver(tol=1e-8, method="BDF")

        def exponential_growth(t, y):
            return y

        def y_eq_5_t_eq_6(t, y):
            return np.concatenate((y - 5, np.array([t]) - 6))

        y0 = np.array([1])
        t_eval = np.linspace(0, 7, 100)
        t_sol, y_sol = solver.integrate(
            exponential_growth, y0, t_eval, events=y_eq_5_t_eq_6
        )
        np.testing.assert_allclose(y_sol[0], np.exp(t_sol), rtol=1e-6)
        np.testing.assert_array_less(t_sol, 6)
        np.testing.assert_array_less(y_sol[0], 5)

    def test_model_solver(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: pybamm.Scalar(0.1) * var}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)
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
