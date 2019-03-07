#
# Tests for the Scikits Solver class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
from tests import StandardModelTest

import unittest
import numpy as np


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestScikitsSolver(unittest.TestCase):
    def test_ode_integrate(self):
        # Constant
        solver = pybamm.ScikitsOdeSolver(tol=1e-8)

        def constant_growth(t, y):
            return 0.5 * np.ones_like(y)

        y0 = np.array([0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(constant_growth, y0, t_eval)
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])

        # Exponential decay
        solver = pybamm.ScikitsOdeSolver(tol=1e-8)

        def exponential_decay(t, y):
            return -0.1 * y

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(exponential_decay, y0, t_eval)
        np.testing.assert_allclose(y_sol[0], np.exp(-0.1 * t_sol))

    def test_ode_integrate_with_event(self):
        # Constant
        solver = pybamm.ScikitsOdeSolver(tol=1e-8)

        def constant_decay(t, y):
            return -2 * np.ones_like(y)

        def y_equal_0(t, y):
            return y[0]

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(constant_decay, y0, t_eval, events=[y_equal_0])
        np.testing.assert_allclose(1 - 2 * t_sol, y_sol[0])
        self.assertLess(len(t_sol), len(t_eval))
        np.testing.assert_array_less(0, y_sol[0])
        np.testing.assert_array_less(t_sol, 0.5)

        # Expnonential growth
        solver = pybamm.ScikitsOdeSolver(tol=1e-8)

        def exponential_growth(t, y):
            return y

        def y_eq_9(t, y):
            return y - 9

        def ysq_eq_7(t, y):
            return y ** 2 - 7

        y0 = np.array([1])
        t_eval = np.linspace(0, 3, 100)
        t_sol, y_sol = solver.integrate(
            exponential_growth, y0, t_eval, events=[y_eq_9, ysq_eq_7]
        )
        self.assertLess(len(t_sol), len(t_eval))
        np.testing.assert_allclose(np.exp(t_sol), y_sol[0], rtol=1e-4)
        np.testing.assert_array_less(y_sol, 9)
        np.testing.assert_array_less(y_sol ** 2, 7)

    def test_dae_integrate(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]]

        y0 = np.array([0, 0])
        ydot0 = np.array([0.5, 1.0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(constant_growth_dae, y0, ydot0, t_eval)
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])
        np.testing.assert_allclose(1.0 * t_sol, y_sol[1])

        # Exponential decay
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return [-0.1 * y[0] - ydot[0], 2 * y[0] - y[1]]

        y0 = np.array([1, 2])
        ydot0 = np.array([-0.1, -0.2])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(exponential_decay_dae, y0, ydot0, t_eval)
        np.testing.assert_allclose(y_sol[0], np.exp(-0.1 * t_sol))
        np.testing.assert_allclose(y_sol[1], 2 * np.exp(-0.1 * t_sol))

    def test_dae_integrate_with_event(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]]

        def y0_eq_2(t, y):
            return y[0] - 2

        def y1_eq_5(t, y):
            return y[1] - 5

        y0 = np.array([0, 0])
        ydot0 = np.array([0.5, 1.0])
        t_eval = np.linspace(0, 7, 100)
        t_sol, y_sol = solver.integrate(
            constant_growth_dae, y0, ydot0, t_eval, events=[y0_eq_2, y1_eq_5]
        )
        self.assertLess(len(t_sol), len(t_eval))
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])
        np.testing.assert_allclose(1.0 * t_sol, y_sol[1])
        np.testing.assert_array_less(y_sol[0], 2)
        np.testing.assert_array_less(y_sol[1], 5)

        # Exponential decay
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return [-0.1 * y[0] - ydot[0], 2 * y[0] - y[1]]

        def y0_eq_0pt9(t, y):
            return y[0] - 0.9

        def t_eq_0pt5(t, y):
            return t - 0.5

        y0 = np.array([1, 2])
        ydot0 = np.array([-0.1, -0.2])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(
            exponential_decay_dae, y0, ydot0, t_eval, events=[y0_eq_0pt9, t_eq_0pt5]
        )

        self.assertLess(len(t_sol), len(t_eval))
        np.testing.assert_allclose(y_sol[0], np.exp(-0.1 * t_sol))
        np.testing.assert_allclose(y_sol[1], 2 * np.exp(-0.1 * t_sol))
        np.testing.assert_array_less(0.9, y_sol[0])
        np.testing.assert_array_less(t_sol, 0.5)

    def test_model_solver_ode(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        model.initial_conditions_ydot = {var: 0.1}
        disc = StandardModelTest(model).disc
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsOdeSolver(tol=1e-9)
        t_eval = np.linspace(0, 1, 100)
        solver.solve(model, t_eval)
        np.testing.assert_array_equal(solver.t, t_eval)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))

    def test_model_solver_ode_events(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        model.initial_conditions_ydot = {var: 0.1}
        model.events = [pybamm.Function(np.min, var - 1.5)]
        disc = StandardModelTest(model).disc
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsOdeSolver(tol=1e-9)
        t_eval = np.linspace(0, 10, 100)
        solver.solve(model, t_eval)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))
        np.testing.assert_array_less(solver.y[0], 1.5)

    def test_model_solver_dae(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.initial_conditions_ydot = {var1: 0.1, var2: 0.2}
        disc = StandardModelTest(model).disc
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solver.solve(model, t_eval)
        np.testing.assert_array_equal(solver.t, t_eval)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))
        np.testing.assert_allclose(solver.y[-1], 2 * np.exp(0.1 * solver.t))

    def test_model_solver_dae_events(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.initial_conditions_ydot = {var1: 0.1, var2: 0.2}
        model.events = [
            pybamm.Function(np.min, var1 - 1.5),
            pybamm.Function(np.min, var2 - 2.5),
        ]
        disc = StandardModelTest(model).disc
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solver.solve(model, t_eval)
        np.testing.assert_array_less(solver.y[0], 1.5)
        np.testing.assert_array_less(solver.y[-1], 2.5)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))
        np.testing.assert_allclose(solver.y[-1], 2 * np.exp(0.1 * solver.t))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
