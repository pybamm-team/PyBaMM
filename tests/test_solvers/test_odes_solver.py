#
# Tests for the Odes Solver class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest
import numpy as np


class TestOdesSolver(unittest.TestCase):
    def test_ode_integrate(self):
        # Constant
        solver = pybamm.OdesOdeSolver(tol=1e-8)

        def constant_growth(t, y):
            return 0.5 * np.ones_like(y)

        y0 = np.array([0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(constant_growth, y0, t_eval)
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])

        # Exponential decay
        solver = pybamm.OdesOdeSolver(tol=1e-8)

        def exponential_decay(t, y):
            return -0.1 * y

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(exponential_decay, y0, t_eval)
        np.testing.assert_allclose(y_sol[0], np.exp(-0.1 * t_sol))

    def test_dae_integrate(self):
        # Constant
        solver = pybamm.OdesDaeSolver(tol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [
                0.5 * np.ones_like(y[0]) - ydot[0],
                2*y[0] - y[1],
            ]

        y0 = np.array([0, 0])
        ydot0 = np.array([0.5, 1.0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(constant_growth_dae, y0, ydot0, t_eval)
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])
        np.testing.assert_allclose(1.0 * t_sol, y_sol[1])

        # Exponential decay
        solver = pybamm.OdesDaeSolver(tol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return [
                -0.1 * y[0] - ydot[0],
                2*y[0] - y[1],
            ]

        y0 = np.array([1, 2])
        ydot0 = np.array([-0.1, -0.2])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(
            exponential_decay_dae, y0, ydot0, t_eval)
        np.testing.assert_allclose(y_sol[0], np.exp(-0.1 * t_sol))
        np.testing.assert_allclose(y_sol[1], 2*np.exp(-0.1 * t_sol))

    def test_model_solver_ode(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var", domain=["whole cell"])
        model.rhs = {var: pybamm.Scalar(0.1) * var}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        # No need to set parameters; can use base discretisation (no spatial operators)
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param)
        disc = pybamm.BaseDiscretisation(mesh)
        disc.process_model(model)

        # Solve
        solver = pybamm.OdesOdeSolver(tol=1e-9)
        t_eval = mesh["time"]
        solver.solve(model, t_eval)
        np.testing.assert_array_equal(solver.t, t_eval)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))

    def test_model_solver_dae(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1", domain=["whole cell"])
        var2 = pybamm.Variable("var2", domain=["whole cell"])
        model.rhs = {var1: pybamm.Scalar(0.1) * var1}
        model.algebraic = [2*var1 - var2]
        model.initial_conditions = {
            var1: pybamm.Scalar(1), var2: pybamm.Scalar(1)}
        # No need to set parameters; can use base discretisation (no spatial operators)
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param)
        disc = pybamm.BaseDiscretisation(mesh)
        disc.process_model(model)

        # Solve
        solver = pybamm.OdesDaeSolver(tol=1e-8)
        t_eval = mesh["time"]
        solver.solve(model, t_eval)
        np.testing.assert_array_equal(solver.t, t_eval)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
