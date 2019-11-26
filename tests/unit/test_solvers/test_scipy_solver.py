#
# Tests for the Scipy Solver class
#
import pybamm
import unittest
import numpy as np
from tests import get_mesh_for_testing
import warnings


class TestScipySolver(unittest.TestCase):
    def test_integrate(self):
        # Constant
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")

        def constant_growth(t, y):
            return 0.5 * np.ones_like(y)

        y0 = np.array([0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(constant_growth, y0, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])

        # Exponential decay
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="BDF")

        def exponential_decay(t, y):
            return -0.1 * y

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(exponential_decay, y0, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        self.assertEqual(solution.termination, "final time")

    def test_integrate_failure(self):
        # Turn off warnings to ignore sqrt error
        warnings.simplefilter("ignore")

        def sqrt_decay(t, y):
            return -np.sqrt(y)

        y0 = np.array([1])
        t_eval = np.linspace(0, 3, 100)
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        # Expect solver to fail when y goes negative
        with self.assertRaises(pybamm.SolverError):
            solver.integrate(sqrt_decay, y0, t_eval)

        # Turn warnings back on
        warnings.simplefilter("default")

    def test_integrate_with_event(self):
        # Constant
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")

        def constant_growth(t, y):
            return 0.5 * np.ones_like(y)

        def y_eq_2(t, y):
            return y - 2

        y0 = np.array([0])
        t_eval = np.linspace(0, 10, 100)
        solution = solver.integrate(constant_growth, y0, t_eval, events=[y_eq_2])
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(solution.t_event, 4)
        np.testing.assert_allclose(solution.y_event, 2)
        self.assertEqual(solution.termination, "event")

        # Exponential decay
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="BDF")

        def exponential_growth(t, y):
            return y

        def y_eq_5(t, y):
            return np.max(y - 5)

        def t_eq_6(t, y):
            return t - 6

        y0 = np.array([1, 2])
        t_eval = np.linspace(0, 7, 100)
        solution = solver.integrate(
            exponential_growth, y0, t_eval, events=[y_eq_5, t_eq_6]
        )
        np.testing.assert_allclose(solution.y[0], np.exp(solution.t), rtol=1e-6)
        np.testing.assert_allclose(solution.y[1], 2 * np.exp(solution.t), rtol=1e-6)
        np.testing.assert_array_less(solution.t, 6)
        np.testing.assert_array_less(solution.y, 5)
        np.testing.assert_allclose(solution.t_event, np.log(5 / 2), rtol=1e-6)
        np.testing.assert_allclose(solution.y_event[0], 5 / 2, rtol=1e-6)
        np.testing.assert_allclose(solution.y_event[1], 5, rtol=1e-6)

    def test_ode_integrate_with_jacobian(self):
        # Linear
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="BDF")

        def linear_ode(t, y):
            return np.array([0.5 * np.ones_like(y[0]), 2.0 - y[0]])

        def jacobian(t, y):
            return np.array([[0.0, 0.0], [-1.0, 0.0]])

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(linear_ode, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(
            2.0 * solution.t - 0.25 * solution.t ** 2, solution.y[1], rtol=1e-4
        )

        # Nonlinear exponential grwoth
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="BDF")

        def exponential_growth(t, y):
            return np.array([y[0], (1.0 - y[0]) * y[1]])

        def jacobian(t, y):
            return np.array([[1.0, 0.0], [-y[1], 1 - y[0]]])

        y0 = np.array([1.0, 1.0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(exponential_growth, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(np.exp(solution.t), solution.y[0], rtol=1e-4)
        np.testing.assert_allclose(
            np.exp(1 + solution.t - np.exp(solution.t)), solution.y[1], rtol=1e-4
        )

    def test_model_solver_python(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))

        # Test time
        self.assertEqual(
            solution.total_time, solution.solve_time + solution.set_up_time
        )

    def test_model_solver_with_event_python(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -0.1 * var}
        model.initial_conditions = {var: 1}
        model.events = {"var=0.5": pybamm.min(var - 0.5)}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval)
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_array_equal(solution.t, t_eval[: len(solution.t)])
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))

    def test_model_solver_ode_with_jacobian_python(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: var1, var2: 1 - var1}
        model.initial_conditions = {var1: 1.0, var2: -1.0}
        model.variables = {"var1": var1, "var2": var2}

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
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

        # Solve
        solver = pybamm.ScipySolver(rtol=1e-9, atol=1e-9)
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

    def test_model_step_python(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")

        # Step once
        dt = 0.1
        step_sol = solver.step(model, dt)
        np.testing.assert_array_equal(step_sol.t, [0, dt])
        np.testing.assert_allclose(step_sol.y[0], np.exp(0.1 * step_sol.t))

        # Step again (return 5 points)
        step_sol_2 = solver.step(model, dt, npts=5)
        np.testing.assert_array_equal(step_sol_2.t, np.linspace(dt, 2 * dt, 5))
        np.testing.assert_allclose(step_sol_2.y[0], np.exp(0.1 * step_sol_2.t))

        # append solutions
        step_sol.append(step_sol_2)

        # Check steps give same solution as solve
        t_eval = step_sol.t
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(solution.y[0], step_sol.y[0])

    def test_model_solver_with_event_with_casadi(self):
        # Create model
        model = pybamm.BaseModel()
        for use_jacobian in [True, False]:
            model.use_jacobian = use_jacobian
            model.convert_to_format = "casadi"
            domain = ["negative electrode", "separator", "positive electrode"]
            var = pybamm.Variable("var", domain=domain)
            model.rhs = {var: -0.1 * var}
            model.initial_conditions = {var: 1}
            model.events = {"var=0.5": pybamm.min(var - 0.5)}
            # No need to set parameters; can use base discretisation (no spatial
            # operators)

            # create discretisation
            mesh = get_mesh_for_testing()
            spatial_methods = {"macroscale": pybamm.FiniteVolume()}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc.process_model(model)
            # Solve
            solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
            t_eval = np.linspace(0, 10, 100)
            solution = solver.solve(model, t_eval)
            self.assertLess(len(solution.t), len(t_eval))
            np.testing.assert_array_equal(solution.t, t_eval[: len(solution.t)])
            np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))

    def test_model_solver_with_inputs_with_casadi(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "casadi"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 1}
        model.events = {"var=0.5": pybamm.min(var - 0.5)}
        # No need to set parameters; can use base discretisation (no spatial
        # operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval, inputs={"rate": 0.1})
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_array_equal(solution.t, t_eval[: len(solution.t)])
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
