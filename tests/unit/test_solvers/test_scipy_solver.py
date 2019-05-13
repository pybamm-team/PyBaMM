#
# Tests for the Scipy Solver class
#
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

        def y_eq_2(t, y):
            return y - 2

        y0 = np.array([0])
        t_eval = np.linspace(0, 10, 100)
        t_sol, y_sol = solver.integrate(constant_growth, y0, t_eval, events=[y_eq_2])
        self.assertLess(len(t_sol), len(t_eval))
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])

        # Exponential decay
        solver = pybamm.ScipySolver(tol=1e-8, method="BDF")

        def exponential_growth(t, y):
            return y

        def y_eq_5(t, y):
            return np.max(y - 5)

        def t_eq_6(t, y):
            return t - 6

        y0 = np.array([1, 2])
        t_eval = np.linspace(0, 7, 100)
        t_sol, y_sol = solver.integrate(
            exponential_growth, y0, t_eval, events=[y_eq_5, t_eq_6]
        )
        np.testing.assert_allclose(y_sol[0], np.exp(t_sol), rtol=1e-6)
        np.testing.assert_allclose(y_sol[1], 2 * np.exp(t_sol), rtol=1e-6)
        np.testing.assert_array_less(t_sol, 6)
        np.testing.assert_array_less(y_sol, 5)

    def test_ode_integrate_with_jacobian(self):
        # Linear
        solver = pybamm.ScipySolver(tol=1e-8, method="BDF")

        def linear_ode(t, y):
            return np.array([0.5 * np.ones_like(y[0]), 2.0 - y[0]])

        def jacobian(t, y):
            return np.array([[0.0, 0.0], [-1.0, 0.0]])

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(linear_ode, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])
        np.testing.assert_allclose(2.0 * t_sol - 0.25 * t_sol ** 2, y_sol[1], rtol=1e-4)

        # Nonlinear exponential grwoth
        solver = pybamm.ScipySolver(tol=1e-8, method="BDF")

        def exponential_growth(t, y):
            return np.array([y[0], (1.0 - y[0]) * y[1]])

        def jacobian(t, y):
            return np.array([[1.0, 0.0], [-y[1], 1 - y[0]]])

        y0 = np.array([1.0, 1.0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(
            exponential_growth, y0, t_eval, jacobian=jacobian
        )
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(np.exp(t_sol), y_sol[0], rtol=1e-4)
        np.testing.assert_allclose(
            np.exp(1 + t_sol - np.exp(t_sol)), y_sol[1], rtol=1e-4
        )

    def test_model_solver(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
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

    def test_model_solver_with_event(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -0.1 * var}
        model.initial_conditions = {var: 1}
        model.events = [pybamm.Function(np.min, var - 0.5)]
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.ScipySolver(tol=1e-8, method="RK45")
        t_eval = np.linspace(0, 10, 100)
        solver.solve(model, t_eval)
        self.assertLess(len(solver.t), len(t_eval))
        np.testing.assert_array_equal(solver.t, t_eval[: len(solver.t)])
        np.testing.assert_allclose(solver.y[0], np.exp(-0.1 * solver.t))

    def test_model_solver_ode_with_jacobian(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: var1, var2: 1 - var1}
        model.initial_conditions = {var1: 1.0, var2: -1.0}
        model.variables = {"var1": var1, "var2": var2}

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # Add user-supplied Jacobian to model
        combined_submesh = mesh.combine_submeshes(
            "negative electrode", "separator", "positive electrode"
        )
        N = combined_submesh[0].npts

        # construct jacobian in order of model.rhs
        J = []
        for var in model.rhs.keys():
            if var.id == var1.id:
                J.append([np.eye(N), np.zeros((N, N))])
            else:
                J.append([-1.0 * np.eye(N), np.zeros((N, N))])

        J = np.block(J)

        def jacobian(t, y):
            return J

        model.jacobian = jacobian

        # Solve
        solver = pybamm.ScipySolver(tol=1e-9)
        t_eval = np.linspace(0, 1, 100)
        solver.solve(model, t_eval)
        np.testing.assert_array_equal(solver.t, t_eval)

        T, Y = solver.t, solver.y
        np.testing.assert_array_almost_equal(
            model.variables["var1"].evaluate(T, Y),
            np.ones((N, T.size)) * np.exp(T[np.newaxis, :]),
        )
        np.testing.assert_array_almost_equal(
            model.variables["var2"].evaluate(T, Y),
            np.ones((N, T.size)) * (T[np.newaxis, :] - np.exp(T[np.newaxis, :])),
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
