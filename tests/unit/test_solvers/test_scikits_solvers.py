#
# Tests for the Scikits Solver classes
#
import pybamm
import numpy as np
import unittest
import warnings
from tests import get_mesh_for_testing, get_discretisation_for_testing


@unittest.skipIf(not pybamm.have_scikits_odes(), "scikits.odes not installed")
class TestScikitsSolvers(unittest.TestCase):
    def test_model_ode_integrate_failure(self):
        # Turn off warnings to ignore sqrt error
        warnings.simplefilter("ignore")

        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: -pybamm.sqrt(var)}
        model.initial_conditions = {var: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 3, 100)
        solver = pybamm.ScikitsOdeSolver()
        # Expect solver to fail when y goes negative
        with self.assertRaises(pybamm.SolverError):
            solver.solve(model, t_eval)

        # Turn warnings back on
        warnings.simplefilter("default")

    def test_model_dae_integrate_failure_bad_ics(self):
        # Force model to fail by providing bad ics
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        # Create custom model so that custom ics
        class Model:
            mass_matrix = pybamm.Matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
            y0 = np.array([0.0, 1.0])
            events_eval = []

            def residuals_eval(self, t, y, ydot):
                return np.array([0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]])

            def jacobian_eval(self, t, y):
                return np.array([[0.0, 0.0], [2.0, -1.0]])

        model = Model()
        t_eval = np.linspace(0, 1, 100)

        with self.assertRaises(pybamm.SolverError):
            solver._integrate(model, t_eval)

    def test_dae_integrate_bad_ics(self):
        # Make sure that dae solver can fix bad ics automatically
        # Constant
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        var2 = pybamm.Variable("var2")
        model.rhs = {var: 0.5}
        model.algebraic = {var2: 2 * var - var2}
        model.initial_conditions = {var: 0, var2: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 1, 100)
        solver.set_up(model)
        # check y0
        np.testing.assert_array_equal(model.y0, [0, 0])
        # check dae solutions
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(1.0 * solution.t, solution.y[1])

    def test_dae_integrate_with_non_unity_mass(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        # Create custom model so that custom mass matrix can be used
        class Model:
            mass_matrix = pybamm.Matrix(np.array([[4.0, 0.0], [0.0, 0.0]]))
            y0 = np.array([0.0, 0.0])
            events_eval = []

            def residuals_eval(self, t, y, ydot):
                return np.array(
                    [0.5 * np.ones_like(y[0]) - 4 * ydot[0], 2.0 * y[0] - y[1]]
                )

            def jacobian_eval(self, t, y):
                return np.array([[0.0, 0.0], [2.0, -1.0]])

        model = Model()
        t_eval = np.linspace(0, 1, 100)
        solution = solver._integrate(model, t_eval)
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
        model.events = {
            "2 * var = 2.5": pybamm.min(2 * var - 2.5),
            "var = 1.5": pybamm.min(var - 1.5),
        }
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
        model.events = {
            "var1 = 1.5": pybamm.min(var1 - 1.5),
            "var2 = 2.5": pybamm.min(var2 - 2.5),
        }
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
        step_sol = solver.step(None, model, dt)
        np.testing.assert_array_equal(step_sol.t, [0, dt])
        np.testing.assert_allclose(step_sol.y[0], np.exp(0.1 * step_sol.t))

        # Step again (return 5 points)
        step_sol_2 = solver.step(step_sol, model, dt, npts=5)
        np.testing.assert_array_equal(
            step_sol_2.t, np.concatenate([np.array([0]), np.linspace(dt, 2 * dt, 5)])
        )
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
        step_sol = solver.step(None, model, dt)
        np.testing.assert_array_equal(step_sol.t, [0, dt])
        np.testing.assert_allclose(step_sol.y[0], np.exp(0.1 * step_sol.t))
        np.testing.assert_allclose(step_sol.y[-1], 2 * np.exp(0.1 * step_sol.t))

        # Step again (return 5 points)
        step_sol_2 = solver.step(step_sol, model, dt, npts=5)
        np.testing.assert_array_equal(
            step_sol_2.t, np.concatenate([np.array([0]), np.linspace(dt, 2 * dt, 5)])
        )
        np.testing.assert_allclose(step_sol_2.y[0], np.exp(0.1 * step_sol_2.t))
        np.testing.assert_allclose(step_sol_2.y[-1], 2 * np.exp(0.1 * step_sol_2.t))

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
        model.events = {
            "2 * var = 2.5": pybamm.min(2 * var - 2.5),
            "var = 1.5": pybamm.min(var - 1.5),
        }
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
            model.events = {
                "var1 = 1.5": pybamm.min(var1 - 1.5),
                "var2 = 2.5": pybamm.min(var2 - 2.5),
            }
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
            model.events = {
                "var1 = 1.5": pybamm.min(var1 - 1.5),
                "var2 = 2.5": pybamm.min(var2 - 2.5),
            }
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

    def test_model_solver_dae_with_external(self):
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
