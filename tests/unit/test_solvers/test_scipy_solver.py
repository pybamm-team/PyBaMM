#
# Tests for the Scipy Solver class
#
import pybamm
import unittest
import numpy as np
from tests import get_mesh_for_testing
import warnings
import sys


class TestScipySolver(unittest.TestCase):
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
        # Make sure that passing in extra options works
        solver = pybamm.ScipySolver(
            rtol=1e-8, atol=1e-8, method="RK45", extra_options={"first_step": 1e-4}
        )
        t_eval = np.linspace(0, 1, 80)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))

        # Test time
        self.assertEqual(
            solution.total_time, solution.solve_time + solution.set_up_time
        )
        self.assertEqual(solution.termination, "final time")

    def test_model_solver_failure(self):
        # Turn off warnings to ignore sqrt error
        warnings.simplefilter("ignore")
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        var = pybamm.Variable("var")
        model.rhs = {var: -pybamm.sqrt(var)}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 3, 100)
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        # Expect solver to fail when y goes negative
        with self.assertRaises(pybamm.SolverError):
            solver.solve(model, t_eval)

        # Turn warnings back on
        warnings.simplefilter("default")

    def test_model_solver_with_event_python(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -0.1 * var}
        model.initial_conditions = {var: 1}
        # needs to work with multiple events (to avoid bug where only last event is
        # used)
        model.events = [
            pybamm.Event("var=0.5", pybamm.min(var - 0.5)),
            pybamm.Event("var=-0.5", pybamm.min(var + 0.5)),
        ]
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
        dt = 1
        step_sol = solver.step(None, model, dt)
        np.testing.assert_array_equal(step_sol.t, [0, dt])
        np.testing.assert_array_almost_equal(step_sol.y[0], np.exp(0.1 * step_sol.t))

        # Step again (return 5 points)
        step_sol_2 = solver.step(step_sol, model, dt, npts=5)
        np.testing.assert_array_equal(
            step_sol_2.t, np.concatenate([np.array([0]), np.linspace(dt, 2 * dt, 5)])
        )
        np.testing.assert_array_almost_equal(
            step_sol_2.y[0], np.exp(0.1 * step_sol_2.t)
        )

        # Check steps give same solution as solve
        t_eval = step_sol.t
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_almost_equal(solution.y[0], step_sol.y[0])

    def test_model_solver_with_inputs(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 1}
        model.events = [pybamm.Event("var=0.5", pybamm.min(var - 0.5))]
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

    def test_model_solver_with_external(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=domain)
        var2 = pybamm.Variable("var2", domain=domain)
        model.rhs = {var1: -var2}
        model.initial_conditions = {var1: 1}
        model.external_variables = [var2]
        model.variables = {"var2": var2}
        # No need to set parameters; can use base discretisation (no spatial
        # operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(
            model, t_eval, external_variables={"var2": 0.5 * np.ones(100)}
        )
        np.testing.assert_allclose(solution.y[0], 1 - 0.5 * solution.t, rtol=1e-06)

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
            # needs to work with multiple events (to avoid bug where only last event is
            # used)
            model.events = [
                pybamm.Event("var=0.5", pybamm.min(var - 0.5)),
                pybamm.Event("var=-0.5", pybamm.min(var + 0.5)),
            ]
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
        model.events = [pybamm.Event("var=0.5", pybamm.min(var - 0.5))]
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

    def test_model_solver_inputs_in_initial_conditions(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        model.rhs = {var1: pybamm.InputParameter("rate") * var1}
        model.initial_conditions = {
            var1: pybamm.InputParameter("ic 1"),
        }

        # Solve
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval, inputs={"rate": -1, "ic 1": 0.1})
        np.testing.assert_array_almost_equal(
            solution.y[0], 0.1 * np.exp(-solution.t), decimal=5
        )

        # Solve again with different initial conditions
        solution = solver.solve(model, t_eval, inputs={"rate": -0.1, "ic 1": 1})
        np.testing.assert_array_almost_equal(
            solution.y[0], 1 * np.exp(-0.1 * solution.t), decimal=5
        )


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
