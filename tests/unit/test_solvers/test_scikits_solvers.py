#
# Tests for the Scikits Solver classes
#
import pybamm
import numpy as np
import scipy.sparse as sparse
import unittest
import warnings
from tests import get_mesh_for_testing, get_discretisation_for_testing

# TODO: remove this
import matplotlib.pylab as plt


@unittest.skipIf(not pybamm.have_scikits_odes(), "scikits.odes not installed")
class TestScikitsSolvers(unittest.TestCase):
    def test_ode_integrate(self):
        # Constant
        solver = pybamm.ScikitsOdeSolver(rtol=1e-8, atol=1e-8)

        def constant_growth(t, y):
            return 0.5 * np.ones_like(y)

        y0 = np.array([0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(constant_growth, y0, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])

        # Exponential decay
        solver = pybamm.ScikitsOdeSolver(rtol=1e-8, atol=1e-8)

        def exponential_decay(t, y):
            return -0.1 * y

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(exponential_decay, y0, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        self.assertEqual(solution.termination, "final time")

    def test_ode_integrate_failure(self):
        # Turn off warnings to ignore sqrt error
        warnings.simplefilter("ignore")

        def sqrt_decay(t, y):
            return -np.sqrt(y)

        y0 = np.array([1])
        t_eval = np.linspace(0, 3, 100)
        solver = pybamm.ScikitsOdeSolver()
        # Expect solver to fail when y goes negative
        with self.assertRaises(pybamm.SolverError):
            solver.integrate(sqrt_decay, y0, t_eval)

        # Turn warnings back on
        warnings.simplefilter("default")

    def test_ode_integrate_with_event(self):
        # Constant
        solver = pybamm.ScikitsOdeSolver(rtol=1e-8, atol=1e-8)

        def constant_decay(t, y):
            return -2 * np.ones_like(y)

        def y_equal_0(t, y):
            return y[0]

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(constant_decay, y0, t_eval, events=[y_equal_0])
        np.testing.assert_allclose(1 - 2 * solution.t, solution.y[0])
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_array_less(0, solution.y[0])
        np.testing.assert_array_less(solution.t, 0.5)
        np.testing.assert_allclose(solution.t_event, 0.5)
        np.testing.assert_allclose(solution.y_event, 0)
        self.assertEqual(solution.termination, "event")

        # Expnonential growth
        solver = pybamm.ScikitsOdeSolver(rtol=1e-8, atol=1e-8)

        def exponential_growth(t, y):
            return y

        def y_eq_9(t, y):
            return y - 9

        def ysq_eq_7(t, y):
            return y ** 2 - 7

        y0 = np.array([1])
        t_eval = np.linspace(0, 3, 100)
        solution = solver.integrate(
            exponential_growth, y0, t_eval, events=[ysq_eq_7, y_eq_9]
        )
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_allclose(np.exp(solution.t), solution.y[0], rtol=1e-4)
        np.testing.assert_array_less(solution.y, 9)
        np.testing.assert_array_less(solution.y ** 2, 7)
        np.testing.assert_allclose(solution.t_event, np.log(7) / 2, rtol=1e-4)
        np.testing.assert_allclose(solution.y_event ** 2, 7, rtol=1e-4)
        self.assertEqual(solution.termination, "event")

    def test_ode_integrate_with_jacobian(self):
        # Linear
        solver = pybamm.ScikitsOdeSolver(rtol=1e-8, atol=1e-8)

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
        solution = solver.integrate(linear_ode, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(
            2.0 * solution.t - 0.25 * solution.t ** 2, solution.y[1], rtol=1e-4
        )

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(linear_ode, y0, t_eval, jacobian=sparse_jacobian)

        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            2.0 * solution.t - 0.25 * solution.t ** 2, solution.y[1], rtol=1e-4
        )
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])

        solver = pybamm.ScikitsOdeSolver(rtol=1e-8, atol=1e-8, linsolver="spgmr")

        solution = solver.integrate(linear_ode, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(
            2.0 * solution.t - 0.25 * solution.t ** 2, solution.y[1], rtol=1e-4
        )

        solution = solver.integrate(linear_ode, y0, t_eval, jacobian=sparse_jacobian)

        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            2.0 * solution.t - 0.25 * solution.t ** 2, solution.y[1], rtol=1e-4
        )
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])

        # Nonlinear exponential grwoth
        solver = pybamm.ScikitsOdeSolver(rtol=1e-8, atol=1e-8)

        def exponential_growth(t, y):
            return np.array([y[0], (1.0 - y[0]) * y[1]])

        def jacobian(t, y):
            return np.array([[1.0, 0.0], [-y[1], 1 - y[0]]])

        def sparse_jacobian(t, y):
            return sparse.csr_matrix(jacobian(t, y))

        y0 = np.array([1.0, 1.0])
        t_eval = np.linspace(0, 1, 100)

        solution = solver.integrate(exponential_growth, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(np.exp(solution.t), solution.y[0], rtol=1e-4)
        np.testing.assert_allclose(
            np.exp(1 + solution.t - np.exp(solution.t)), solution.y[1], rtol=1e-4
        )

        solution = solver.integrate(
            exponential_growth, y0, t_eval, jacobian=sparse_jacobian
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(np.exp(solution.t), solution.y[0], rtol=1e-4)
        np.testing.assert_allclose(
            np.exp(1 + solution.t - np.exp(solution.t)), solution.y[1], rtol=1e-4
        )

        solver = pybamm.ScikitsOdeSolver(rtol=1e-8, atol=1e-8, linsolver="spgmr")

        solution = solver.integrate(exponential_growth, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(np.exp(solution.t), solution.y[0], rtol=1e-4)
        np.testing.assert_allclose(
            np.exp(1 + solution.t - np.exp(solution.t)), solution.y[1], rtol=1e-4
        )

        solution = solver.integrate(
            exponential_growth, y0, t_eval, jacobian=sparse_jacobian
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(np.exp(solution.t), solution.y[0], rtol=1e-4)
        np.testing.assert_allclose(
            np.exp(1 + solution.t - np.exp(solution.t)), solution.y[1], rtol=1e-4
        )

    def test_dae_integrate(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]]

        y0 = np.array([0, 0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(constant_growth_dae, y0, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(1.0 * solution.t, solution.y[1])

        # Exponential decay
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return [-0.1 * y[0] - ydot[0], 2 * y[0] - y[1]]

        y0 = np.array([1, 2])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(exponential_decay_dae, y0, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        np.testing.assert_allclose(solution.y[1], 2 * np.exp(-0.1 * solution.t))
        self.assertEqual(solution.termination, "final time")

    def test_dae_integrate_failure(self):
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]]

        y0 = np.array([0, 1])
        t_eval = np.linspace(0, 1, 100)
        with self.assertRaises(pybamm.SolverError):
            solver.integrate(constant_growth_dae, y0, t_eval)

    def test_dae_integrate_bad_ics(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

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
        solution = solver.integrate(constant_growth_dae, y0, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(1.0 * solution.t, solution.y[1])

        # Exponential decay
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return [-0.1 * y[0] - ydot[0], 2 * y[0] - y[1]]

        y0 = np.array([1, 2])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(exponential_decay_dae, y0, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        np.testing.assert_allclose(solution.y[1], 2 * np.exp(-0.1 * solution.t))

    def test_dae_integrate_with_event(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]]

        def y0_eq_2(t, y):
            return y[0] - 2

        def y1_eq_5(t, y):
            return y[1] - 5

        y0 = np.array([0, 0])
        t_eval = np.linspace(0, 7, 100)
        solution = solver.integrate(
            constant_growth_dae, y0, t_eval, events=[y0_eq_2, y1_eq_5]
        )
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(1.0 * solution.t, solution.y[1])
        np.testing.assert_array_less(solution.y[0], 2)
        np.testing.assert_array_less(solution.y[1], 5)
        np.testing.assert_allclose(solution.t_event, 4)
        np.testing.assert_allclose(solution.y_event[0], 2)
        np.testing.assert_allclose(solution.y_event[1], 4)
        self.assertEqual(solution.termination, "event")

        # Exponential decay
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return np.array([-0.1 * y[0] - ydot[0], 2 * y[0] - y[1]])

        def y0_eq_0pt9(t, y):
            return y[0] - 0.9

        def t_eq_0pt5(t, y):
            return t - 0.5

        y0 = np.array([1, 2])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(
            exponential_decay_dae, y0, t_eval, events=[y0_eq_0pt9, t_eq_0pt5]
        )

        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        np.testing.assert_allclose(solution.y[1], 2 * np.exp(-0.1 * solution.t))
        np.testing.assert_array_less(0.9, solution.y[0])
        np.testing.assert_array_less(solution.t, 0.5)
        np.testing.assert_allclose(solution.t_event, 0.5)
        np.testing.assert_allclose(solution.y_event[0], np.exp(-0.1 * 0.5))
        np.testing.assert_allclose(solution.y_event[1], 2 * np.exp(-0.1 * 0.5))
        self.assertEqual(solution.termination, "event")

    def test_dae_integrate_with_jacobian(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return np.array([0.5 * np.ones_like(y[0]) - ydot[0], 2.0 * y[0] - y[1]])

        mass_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])

        def jacobian(t, y):
            return np.array([[0.0, 0.0], [2.0, -1.0]])

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(
            constant_growth_dae, y0, t_eval, mass_matrix=mass_matrix, jacobian=jacobian
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(1.0 * solution.t, solution.y[1])

        # Nonlinear (tests when Jacobian a function of y)
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        def nonlinear_dae(t, y, ydot):
            return np.array([0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] ** 2 - y[1]])

        mass_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])

        def jacobian(t, y):
            return np.array([[0.0, 0.0], [4.0 * y[0], -1.0]])

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(
            nonlinear_dae, y0, t_eval, mass_matrix=mass_matrix, jacobian=jacobian
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(0.5 * solution.t ** 2, solution.y[1])

    def test_dae_integrate_with_non_unity_mass(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return np.array([0.5 * np.ones_like(y[0]) - 4 * ydot[0], 2.0 * y[0] - y[1]])

        mass_matrix = np.array([[4.0, 0.0], [0.0, 0.0]])

        def jacobian(t, y):
            return np.array([[0.0, 0.0], [2.0, -1.0]])

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(
            constant_growth_dae, y0, t_eval, mass_matrix=mass_matrix, jacobian=jacobian
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.125 * solution.t, solution.y[0])
        np.testing.assert_allclose(0.25 * solution.t, solution.y[1])

    def test_model_solver_ode_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsOdeSolver(rtol=1e-9, atol=1e-9)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))

    def test_model_solver_ode_events_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        model.events = [
            pybamm.Event("2 * var = 2.5", pybamm.min(2 * var - 2.5)),
            pybamm.Event("var = 1.5", pybamm.min(var - 1.5)),
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsOdeSolver(rtol=1e-9, atol=1e-9)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_array_less(solution.y[0], 1.5)
        np.testing.assert_array_less(solution.y[0], 1.25)

    def test_model_solver_ode_jacobian_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: var1, var2: 1 - var1}
        model.initial_conditions = {var1: 1.0, var2: -1.0}
        model.variables = {"var1": var1, "var2": var2}

        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Add user-supplied Jacobian to model
        mesh = get_mesh_for_testing()
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

        # Solve
        solver = pybamm.ScikitsOdeSolver(rtol=1e-9, atol=1e-9)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)

        T, Y = solution.t, solution.y
        np.testing.assert_array_almost_equal(
            model.variables["var1"].evaluate(T, Y),
            np.ones((N, T.size)) * np.exp(T[np.newaxis, :]),
        )
        np.testing.assert_array_almost_equal(
            model.variables["var2"].evaluate(T, Y),
            np.ones((N, T.size)) * (T[np.newaxis, :] - np.exp(T[np.newaxis, :])),
        )

    def test_model_solver_dae_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.use_jacobian = False
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_bad_ics_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 3}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_events_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = [
            pybamm.Event("var1 = 1.5", pybamm.min(var1 - 1.5)),
            pybamm.Event("var2 = 2.5", pybamm.min(var2 - 2.5)),
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y[0], 1.5)
        np.testing.assert_array_less(solution.y[-1], 2.5)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_nonsmooth_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)

        def nonsmooth_rate(t):
            return 0.1 * int(t < 2.5) + 0.1
        rate = pybamm.Function(nonsmooth_rate, pybamm.t)
        model.rhs = {var1: rate * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = [
            pybamm.Event("var1 = 1.5", pybamm.min(var1 - 1.5)),
            pybamm.Event("var2 = 2.5", pybamm.min(var2 - 2.5)),
            pybamm.Event("nonsmooth rate",
                         pybamm.Scalar(2.5),
                         pybamm.EventType.DISCONTINUITY
                         )
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        plt.plot(solution.y[0])
        plt.plot(solution.y[1])
        plt.show()
        #np.testing.assert_array_less(solution.y[0], 1.5)
        #np.testing.assert_array_less(solution.y[-1], 2.5)
        #np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        #np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_with_jacobian_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
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
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_solve_ode_model_with_dae_solver_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))

    def test_model_step_ode_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        solver = pybamm.ScikitsOdeSolver(rtol=1e-9, atol=1e-9)

        # Step once
        dt = 0.1
        step_sol = solver.step(model, dt)
        np.testing.assert_array_equal(step_sol.t, [0, dt])
        np.testing.assert_allclose(step_sol.y[0], np.exp(0.1 * step_sol.t))

        # Step again (return 5 points)
        step_sol_2 = solver.step(model, dt, npts=5)
        np.testing.assert_array_equal(step_sol_2.t, np.linspace(dt, 2 * dt, 5))
        np.testing.assert_allclose(step_sol_2.y[0], np.exp(0.1 * step_sol_2.t))

        # Check steps give same solution as solve
        t_eval = np.concatenate((step_sol.t, step_sol_2.t[1:]))
        solution = solver.solve(model, t_eval)
        concatenated_steps = np.concatenate((step_sol.y[0], step_sol_2.y[0, 1:]))
        np.testing.assert_allclose(solution.y[0], concatenated_steps)

    def test_model_step_dae_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.use_jacobian = False
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        # Step once
        dt = 0.1
        step_sol = solver.step(model, dt)
        np.testing.assert_array_equal(step_sol.t, [0, dt])
        np.testing.assert_allclose(step_sol.y[0], np.exp(0.1 * step_sol.t))
        np.testing.assert_allclose(step_sol.y[-1], 2 * np.exp(0.1 * step_sol.t))

        # Step again (return 5 points)
        step_sol_2 = solver.step(model, dt, npts=5)
        np.testing.assert_array_equal(step_sol_2.t, np.linspace(dt, 2 * dt, 5))
        np.testing.assert_allclose(step_sol_2.y[0], np.exp(0.1 * step_sol_2.t))
        np.testing.assert_allclose(step_sol_2.y[-1], 2 * np.exp(0.1 * step_sol_2.t))

        # append solutions
        step_sol.append(step_sol_2)

        # Check steps give same solution as solve
        t_eval = step_sol.t
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(solution.y[0], step_sol.y[0, :])
        np.testing.assert_allclose(solution.y[-1], step_sol.y[-1, :])

    def test_model_solver_ode_events_casadi(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "casadi"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        model.events = [
            pybamm.Event("2 * var = 2.5", pybamm.min(2 * var - 2.5)),
            pybamm.Event("var = 1.5", pybamm.min(var - 1.5)),
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsOdeSolver(rtol=1e-9, atol=1e-9)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_array_less(solution.y[0], 1.5)
        np.testing.assert_array_less(solution.y[0], 1.25)

    def test_model_solver_dae_events_casadi(self):
        # Create model
        model = pybamm.BaseModel()
        for use_jacobian in [True, False]:
            model.use_jacobian = use_jacobian
            model.convert_to_format = "casadi"
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            var1 = pybamm.Variable("var1", domain=whole_cell)
            var2 = pybamm.Variable("var2", domain=whole_cell)
            model.rhs = {var1: 0.1 * var1}
            model.algebraic = {var2: 2 * var1 - var2}
            model.initial_conditions = {var1: 1, var2: 2}
            model.events = [
                pybamm.Event("var1 = 1.5", pybamm.min(var1 - 1.5)),
                pybamm.Event("var2 = 2.5", pybamm.min(var2 - 2.5)),
            ]
            disc = get_discretisation_for_testing()
            disc.process_model(model)

            # Solve
            solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
            t_eval = np.linspace(0, 5, 100)
            solution = solver.solve(model, t_eval)
            np.testing.assert_array_less(solution.y[0], 1.5)
            np.testing.assert_array_less(solution.y[-1], 2.5)
            np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
            np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_inputs_events(self):
        # Create model
        for form in ["python", "casadi"]:
            model = pybamm.BaseModel()
            model.convert_to_format = form
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            var1 = pybamm.Variable("var1", domain=whole_cell)
            var2 = pybamm.Variable("var2", domain=whole_cell)
            model.rhs = {var1: pybamm.InputParameter("rate 1") * var1}
            model.algebraic = {var2: pybamm.InputParameter("rate 2") * var1 - var2}
            model.initial_conditions = {var1: 1, var2: 2}
            model.events = [
                pybamm.Event("var1 = 1.5", pybamm.min(var1 - 1.5)),
                pybamm.Event("var2 = 2.5", pybamm.min(var2 - 2.5)),
            ]
            disc = get_discretisation_for_testing()
            disc.process_model(model)

            # Solve
            solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
            t_eval = np.linspace(0, 5, 100)
            solution = solver.solve(model, t_eval, inputs={"rate 1": 0.1, "rate 2": 2})
            np.testing.assert_array_less(solution.y[0], 1.5)
            np.testing.assert_array_less(solution.y[-1], 2.5)
            np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
            np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae__with_external(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=domain)
        var2 = pybamm.Variable("var2", domain=domain)
        model.rhs = {var1: -var2}
        model.initial_conditions = {var1: 1}
        model.external_variables = [var2]
        model.variables = {"var1": var1, "var2": var2}
        # No need to set parameters; can use base discretisation (no spatial
        # operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval, external_variables={"var2": 0.5})
        np.testing.assert_allclose(solution.y[0], 1 - 0.5 * solution.t, rtol=1e-06)

    def test_solve_ode_model_with_dae_solver_casadi(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "casadi"
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
