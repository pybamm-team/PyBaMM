#
# Tests for the Scikits Solver class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
from tests import get_mesh_for_testing, get_discretisation_for_testing

import unittest
import numpy as np
import scipy.sparse as sparse


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
            exponential_growth, y0, t_eval, events=[ysq_eq_7, y_eq_9]
        )
        self.assertLess(len(t_sol), len(t_eval))
        np.testing.assert_allclose(np.exp(t_sol), y_sol[0], rtol=1e-4)
        np.testing.assert_array_less(y_sol, 9)
        np.testing.assert_array_less(y_sol ** 2, 7)

    def test_ode_integrate_with_jacobian(self):
        # Linear
        solver = pybamm.ScikitsOdeSolver(tol=1e-8)

        def linear_ode(t, y):
            return np.array([0.5, 2 - y[0]])

        J = np.array([[0.0, 0.0], [-1.0, 0.0]])
        sJ = sparse.csr_matrix(J)

        def jacobian(t, y):
            return J

        def sparse_jacobian(t, y):
            return sJ

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(linear_ode, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])
        np.testing.assert_allclose(2.0 * t_sol - 0.25 * t_sol ** 2, y_sol[1], rtol=1e-4)

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(
            linear_ode, y0, t_eval, jacobian=sparse_jacobian)

        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(2.0 * t_sol - 0.25 * t_sol ** 2, y_sol[1], rtol=1e-4)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])

        solver = pybamm.ScikitsOdeSolver(tol=1e-8, linsolver = "spgmr")

        t_sol, y_sol = solver.integrate(linear_ode, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])
        np.testing.assert_allclose(2.0 * t_sol - 0.25 * t_sol ** 2, y_sol[1], rtol=1e-4)

        t_sol, y_sol = solver.integrate(
            linear_ode, y0, t_eval, jacobian=sparse_jacobian)

        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(2.0 * t_sol - 0.25 * t_sol ** 2, y_sol[1], rtol=1e-4)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])


        # Nonlinear exponential grwoth
        solver = pybamm.ScikitsOdeSolver(tol=1e-8)

        def exponential_growth(t, y):
            return np.array([y[0], (1.0 - y[0]) * y[1]])

        def jacobian(t, y):
            return np.array([[1.0, 0.0], [-y[1], 1 - y[0]]])

        def sparse_jacobian(t, y):
            return sparse.csr_matrix(jacobian(t, y))

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

        t_sol, y_sol = solver.integrate(
            exponential_growth, y0, t_eval, jacobian=sparse_jacobian,
        )
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(np.exp(t_sol), y_sol[0], rtol=1e-4)
        np.testing.assert_allclose(
            np.exp(1 + t_sol - np.exp(t_sol)), y_sol[1], rtol=1e-4
        )

        solver = pybamm.ScikitsOdeSolver(tol=1e-8, linsolver="spgmr")

        t_sol, y_sol = solver.integrate(
            exponential_growth, y0, t_eval, jacobian=jacobian
        )
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(np.exp(t_sol), y_sol[0], rtol=1e-4)
        np.testing.assert_allclose(
            np.exp(1 + t_sol - np.exp(t_sol)), y_sol[1], rtol=1e-4
        )

        t_sol, y_sol = solver.integrate(
            exponential_growth, y0, t_eval, jacobian=sparse_jacobian,
        )
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(np.exp(t_sol), y_sol[0], rtol=1e-4)
        np.testing.assert_allclose(
            np.exp(1 + t_sol - np.exp(t_sol)), y_sol[1], rtol=1e-4
        )

    def test_dae_integrate(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]]

        y0 = np.array([0, 0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(constant_growth_dae, y0, t_eval)
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])
        np.testing.assert_allclose(1.0 * t_sol, y_sol[1])

        # Exponential decay
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return [-0.1 * y[0] - ydot[0], 2 * y[0] - y[1]]

        y0 = np.array([1, 2])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(exponential_decay_dae, y0, t_eval)
        np.testing.assert_allclose(y_sol[0], np.exp(-0.1 * t_sol))
        np.testing.assert_allclose(y_sol[1], 2 * np.exp(-0.1 * t_sol))

    def test_dae_integrate_bad_ics(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]]

        def constant_growth_dae_rhs(t, y):
            return np.array([constant_growth_dae(t, y, [0])[0]])

        def constant_growth_dae_algebraic(t, y):
            return np.array([constant_growth_dae(t, y, [0])[1]])

        y0_guess = np.array([0, 1])
        t_eval = np.linspace(0, 1, 100)
        y0 = solver.calculate_consistent_initial_conditions(
            constant_growth_dae_rhs, constant_growth_dae_algebraic, y0_guess
        )
        # check y0
        np.testing.assert_array_equal(y0, [0, 0])
        # check dae solutions
        t_sol, y_sol = solver.integrate(constant_growth_dae, y0, t_eval)
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])
        np.testing.assert_allclose(1.0 * t_sol, y_sol[1])

        # Exponential decay
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return [-0.1 * y[0] - ydot[0], 2 * y[0] - y[1]]

        y0 = np.array([1, 2])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(exponential_decay_dae, y0, t_eval)
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
        t_eval = np.linspace(0, 7, 100)
        t_sol, y_sol = solver.integrate(
            constant_growth_dae, y0, t_eval, events=[y0_eq_2, y1_eq_5]
        )
        self.assertLess(len(t_sol), len(t_eval))
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])
        np.testing.assert_allclose(1.0 * t_sol, y_sol[1])
        np.testing.assert_array_less(y_sol[0], 2)
        np.testing.assert_array_less(y_sol[1], 5)

        # Exponential decay
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return np.array([-0.1 * y[0] - ydot[0], 2 * y[0] - y[1]])

        def y0_eq_0pt9(t, y):
            return y[0] - 0.9

        def t_eq_0pt5(t, y):
            return t - 0.5

        y0 = np.array([1, 2])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(
            exponential_decay_dae, y0, t_eval, events=[y0_eq_0pt9, t_eq_0pt5]
        )

        self.assertLess(len(t_sol), len(t_eval))
        np.testing.assert_allclose(y_sol[0], np.exp(-0.1 * t_sol))
        np.testing.assert_allclose(y_sol[1], 2 * np.exp(-0.1 * t_sol))
        np.testing.assert_array_less(0.9, y_sol[0])
        np.testing.assert_array_less(t_sol, 0.5)

    def test_dae_integrate_with_jacobian(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return np.array([0.5 * np.ones_like(y[0]) - ydot[0], 2.0 * y[0] - y[1]])

        mass_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])

        def jacobian(t, y):
            return np.array([[0.0, 0.0], [2.0, -1.0]])

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(
            constant_growth_dae, y0, t_eval, mass_matrix=mass_matrix, jacobian=jacobian
        )
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])
        np.testing.assert_allclose(1.0 * t_sol, y_sol[1])

        # Nonlinear (tests when Jacobian a function of y)
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)

        def nonlinear_dae(t, y, ydot):
            return np.array([0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] ** 2 - y[1]])

        mass_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])

        def jacobian(t, y):
            return np.array([[0.0, 0.0], [4.0 * y[0], -1.0]])

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        t_sol, y_sol = solver.integrate(
            nonlinear_dae, y0, t_eval, mass_matrix=mass_matrix, jacobian=jacobian
        )
        np.testing.assert_array_equal(t_sol, t_eval)
        np.testing.assert_allclose(0.5 * t_sol, y_sol[0])
        np.testing.assert_allclose(0.5 * t_sol ** 2, y_sol[1])

    def test_model_solver_ode(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        disc = get_discretisation_for_testing()
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
        model.events = [
            pybamm.Function(np.min, 2 * var - 2.5),
            pybamm.Function(np.min, var - 1.5),
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsOdeSolver(tol=1e-9)
        t_eval = np.linspace(0, 10, 100)
        solver.solve(model, t_eval)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))
        np.testing.assert_array_less(solver.y[0], 1.5)
        np.testing.assert_array_less(solver.y[0], 1.25)

    def test_model_solver_ode_jacobian(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: var1, var2: 1 - var1}
        model.initial_conditions = {var1: 1.0, var2: -1.0}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Add user-supplied Jacobian to model
        mesh = get_mesh_for_testing()
        combined_submesh = mesh.combine_submeshes(
            "negative electrode", "separator", "positive electrode"
        )
        N = combined_submesh[0].npts

        def jacobian(t, y):
            return np.block(
                [[np.eye(N), np.zeros((N, N))], [-1.0 * np.eye(N), np.zeros((N, N))]]
            )

        model.jacobian = jacobian

        # Solve
        solver = pybamm.ScikitsOdeSolver(tol=1e-9)
        t_eval = np.linspace(0, 1, 100)
        solver.solve(model, t_eval)
        np.testing.assert_array_equal(solver.t, t_eval)
        np.testing.assert_allclose(solver.y[0], np.exp(solver.t))
        np.testing.assert_allclose(solver.y[-1], solver.t - np.exp(solver.t))

    def test_model_solver_dae(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solver.solve(model, t_eval)
        np.testing.assert_array_equal(solver.t, t_eval)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))
        np.testing.assert_allclose(solver.y[-1], 2 * np.exp(0.1 * solver.t))

    def test_model_solver_dae_bad_ics(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 3}
        disc = get_discretisation_for_testing()
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
        model.events = [
            pybamm.Function(np.min, var1 - 1.5),
            pybamm.Function(np.min, var2 - 2.5),
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solver.solve(model, t_eval)
        np.testing.assert_array_less(solver.y[0], 1.5)
        np.testing.assert_array_less(solver.y[-1], 2.5)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))
        np.testing.assert_allclose(solver.y[-1], 2 * np.exp(0.1 * solver.t))

    def test_model_solver_dae_with_jacobian(self):
        # Create simple test model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1.0, var2: 2.0}
        model.initial_conditions_ydot = {var1: 0.1, var2: 0.2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Add user-supplied Jacobian to model
        mesh = get_mesh_for_testing()
        combined_submesh = mesh.combine_submeshes(
            "negative electrode", "separator", "positive electrode"
        )
        N = combined_submesh[0].npts

        def jacobian(t, y):
            return np.block(
                [
                    [0.1 * np.eye(N), np.zeros((N, N))],
                    [2.0 * np.eye(N), -1.0 * np.eye(N)],
                ]
            )

        model.jacobian = jacobian

        # Solve
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solver.solve(model, t_eval)
        np.testing.assert_array_equal(solver.t, t_eval)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))
        np.testing.assert_allclose(solver.y[-1], 2 * np.exp(0.1 * solver.t))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
