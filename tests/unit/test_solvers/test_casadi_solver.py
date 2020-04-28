#
# Tests for the Casadi Solver class
#
import pybamm
import unittest
import numpy as np
from tests import get_mesh_for_testing, get_discretisation_for_testing
from scipy.sparse import eye


class TestCasadiSolver(unittest.TestCase):
    def test_bad_mode(self):
        with self.assertRaisesRegex(ValueError, "invalid mode"):
            pybamm.CasadiSolver(mode="bad mode")

    def test_model_solver(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)
        # Solve
        solver = pybamm.CasadiSolver(mode="fast", rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_array_almost_equal(
            solution.y[0], np.exp(0.1 * solution.t), decimal=5
        )

        # Safe mode (enforce events that won't be triggered)
        model.events = [pybamm.Event("an event", var + 1)]
        disc.process_model(model)
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_array_almost_equal(
            solution.y[0], np.exp(0.1 * solution.t), decimal=5
        )

    def test_model_solver_python(self):
        # Create model
        pybamm.set_logging_level("ERROR")
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)
        # Solve
        solver = pybamm.CasadiSolver(mode="fast", rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_array_almost_equal(
            solution.y[0], np.exp(0.1 * solution.t), decimal=5
        )
        pybamm.set_logging_level("WARNING")

    def test_model_solver_failure(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: -pybamm.sqrt(var)}
        model.initial_conditions = {var: 1}
        # add events so that safe mode is used (won't be triggered)
        model.events = [pybamm.Event("10", var - 10)]
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.CasadiSolver(extra_options_call={"regularity_check": False})

        # Solve with failure at t=2
        t_eval = np.linspace(0, 20, 100)
        with self.assertRaises(pybamm.SolverError):
            solver.solve(model, t_eval)
        # Solve with failure at t=0
        model.initial_conditions = {var: 0}
        disc.process_model(model)
        t_eval = np.linspace(0, 20, 100)
        with self.assertRaises(pybamm.SolverError):
            solver.solve(model, t_eval)

    def test_model_solver_events(self):
        # Create model
        model = pybamm.BaseModel()
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
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y[0], 1.5)
        np.testing.assert_array_less(solution.y[-1], 2.5)
        np.testing.assert_array_almost_equal(
            solution.y[0], np.exp(0.1 * solution.t), decimal=5
        )
        np.testing.assert_array_almost_equal(
            solution.y[-1], 2 * np.exp(0.1 * solution.t), decimal=5
        )

    def test_model_step(self):
        # Create model
        model = pybamm.BaseModel()
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

        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)

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

    def test_model_step_with_input(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        a = pybamm.InputParameter("a")
        model.rhs = {var: a * var}
        model.initial_conditions = {var: 1}
        model.variables = {"a": a}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)

        # Step with an input
        dt = 0.1
        step_sol = solver.step(None, model, dt, npts=5, inputs={"a": 0.1})
        np.testing.assert_array_equal(step_sol.t, np.linspace(0, dt, 5))
        np.testing.assert_allclose(step_sol.y[0], np.exp(0.1 * step_sol.t))

        # Step again with different inputs
        step_sol_2 = solver.step(step_sol, model, dt, npts=5, inputs={"a": -1})
        np.testing.assert_array_equal(step_sol_2.t, np.linspace(0, 2 * dt, 9))
        np.testing.assert_array_equal(
            step_sol_2["a"].entries, np.array([0.1, 0.1, 0.1, 0.1, 0.1, -1, -1, -1, -1])
        )
        np.testing.assert_allclose(
            step_sol_2.y[0],
            np.concatenate(
                [
                    np.exp(0.1 * step_sol.t[:5]),
                    np.exp(0.1 * step_sol.t[4]) * np.exp(-(step_sol.t[5:] - dt)),
                ]
            ),
        )

    def test_model_step_events(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = [
            pybamm.Event("var1 = 1.5", pybamm.min(var1 - 1.5)),
            pybamm.Event("var2 = 2.5", pybamm.min(var2 - 2.5)),
        ]
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        step_solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        dt = 0.05
        time = 0
        end_time = 5
        step_solution = None
        while time < end_time:
            step_solution = step_solver.step(step_solution, model, dt=dt, npts=10)
            time += dt
        np.testing.assert_array_less(step_solution.y[0], 1.5)
        np.testing.assert_array_less(step_solution.y[-1], 2.5001)
        np.testing.assert_array_almost_equal(
            step_solution.y[0], np.exp(0.1 * step_solution.t), decimal=5
        )
        np.testing.assert_array_almost_equal(
            step_solution.y[-1], 2 * np.exp(0.1 * step_solution.t), decimal=4
        )

    def test_model_solver_with_inputs(self):
        # Create model
        model = pybamm.BaseModel()
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
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval, inputs={"rate": 0.1})
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_array_equal(solution.t, t_eval[: len(solution.t)])
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t), rtol=1e-06)

    def test_model_solver_dae_inputs_in_initial_conditions(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.rhs = {var1: pybamm.InputParameter("rate") * var1}
        model.algebraic = {var2: var1 - var2}
        model.initial_conditions = {
            var1: pybamm.InputParameter("ic 1"),
            var2: pybamm.InputParameter("ic 2"),
        }

        # Solve
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(
            model, t_eval, inputs={"rate": -1, "ic 1": 0.1, "ic 2": 2}
        )
        np.testing.assert_array_almost_equal(
            solution.y[0], 0.1 * np.exp(-solution.t), decimal=5
        )
        np.testing.assert_array_almost_equal(
            solution.y[-1], 0.1 * np.exp(-solution.t), decimal=5
        )

        # Solve again with different initial conditions
        solution = solver.solve(
            model, t_eval, inputs={"rate": -0.1, "ic 1": 1, "ic 2": 3}
        )
        np.testing.assert_array_almost_equal(
            solution.y[0], 1 * np.exp(-0.1 * solution.t), decimal=5
        )
        np.testing.assert_array_almost_equal(
            solution.y[-1], 1 * np.exp(-0.1 * solution.t), decimal=5
        )

    def test_model_solver_with_external(self):
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
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval, external_variables={"var2": 0.5})
        np.testing.assert_allclose(solution.y[0], 1 - 0.5 * solution.t, rtol=1e-06)

    def test_model_solver_with_non_identity_mass(self):
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1", domain="negative electrode")
        var2 = pybamm.Variable("var2", domain="negative electrode")
        model.rhs = {var1: var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # FV discretisation has identity mass. Manually set the mass matrix to
        # be a diag of 10s here for testing. Note that the algebraic part is all
        # zeros
        mass_matrix = 10 * model.mass_matrix.entries
        model.mass_matrix = pybamm.Matrix(mass_matrix)

        # Note that mass_matrix_inv is just the inverse of the ode block of the
        # mass matrix
        mass_matrix_inv = 0.1 * eye(int(mass_matrix.shape[0] / 2))
        model.mass_matrix_inv = pybamm.Matrix(mass_matrix_inv)

        # Solve
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_dae_solver_algebraic_model(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.algebraic = {var: var + 1}
        model.initial_conditions = {var: 0}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.CasadiSolver()
        t_eval = np.linspace(0, 1)
        with self.assertRaisesRegex(
            pybamm.SolverError, "Cannot use CasadiSolver to solve algebraic model"
        ):
            solver.solve(model, t_eval)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
