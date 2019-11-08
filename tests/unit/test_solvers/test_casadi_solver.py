#
# Tests for the Casadi Solver class
#
import casadi
import pybamm
import unittest
import numpy as np
from tests import get_mesh_for_testing, get_discretisation_for_testing
import warnings


class TestCasadiSolver(unittest.TestCase):
    def test_integrate(self):
        # Constant
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="idas")

        y = casadi.MX.sym("y")
        constant_growth = casadi.MX(0.5)
        problem = {"x": y, "ode": constant_growth}

        y0 = np.array([0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate_casadi(problem, y0, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])

        # Exponential decay
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="cvodes")

        exponential_decay = -0.1 * y
        problem = {"x": y, "ode": exponential_decay}

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate_casadi(problem, y0, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        self.assertEqual(solution.termination, "final time")

    def test_integrate_failure(self):
        # Turn off warnings to ignore sqrt error
        warnings.simplefilter("ignore")

        y = casadi.MX.sym("y")
        sqrt_decay = -np.sqrt(y)

        y0 = np.array([1])
        t_eval = np.linspace(0, 3, 100)
        solver = pybamm.CasadiSolver()
        problem = {"x": y, "ode": sqrt_decay}
        # Expect solver to fail when y goes negative
        with self.assertRaises(pybamm.SolverError):
            solver.integrate_casadi(problem, y0, t_eval)

        # Set up as a model and solve
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.Function(np.sqrt, var)}
        model.initial_conditions = {var: 1}
        # add events so that safe mode is used (won't be triggered)
        model.events = {"10": var - 10}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve with failure at t=2
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="idas")
        t_eval = np.linspace(0, 20, 100)
        with self.assertRaises(pybamm.SolverError):
            solver.solve(model, t_eval)
        # Solve with failure at t=0
        model.initial_conditions = {var: 0}
        disc.process_model(model)
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="idas")
        t_eval = np.linspace(0, 20, 100)
        with self.assertRaises(pybamm.SolverError):
            solver.solve(model, t_eval)

        # Turn warnings back on
        warnings.simplefilter("default")

    def test_bad_mode(self):
        with self.assertRaisesRegex(ValueError, "invalid mode"):
            pybamm.CasadiSolver(mode="bad mode")

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
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="idas")
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))

        # Safe mode (enforce events that won't be triggered)
        model.events = {"an event": var + 1}
        disc.process_model(model)
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="idas")
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))

    def test_model_solver_events(self):
        # Create model
        model = pybamm.BaseModel()
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
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="idas")

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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
