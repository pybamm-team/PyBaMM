#
# Tests for the Casadi Solver class
#
import casadi
import pybamm
import unittest
import numpy as np
from tests import get_mesh_for_testing, get_discretisation_for_testing
from scipy.sparse import eye


class TestCasadiSolver(unittest.TestCase):
    def test_integrate(self):
        # Constant
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="idas")

        t = casadi.MX.sym("t")
        y = casadi.MX.sym("y")
        u = casadi.MX.sym("u")
        constant_growth = casadi.MX(0.5)

        class ConstantGrowthModel:
            casadi_rhs = casadi.Function("rhs", [t, y, u], [constant_growth])
            casadi_algebraic = None
            y0 = np.array([0])

        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(ConstantGrowthModel(), t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])

        # Exponential decay
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="cvodes")

        exponential_decay = -0.1 * y

        class ExponentialDecayModel:
            casadi_rhs = casadi.Function("rhs", [t, y, u], [exponential_decay])
            casadi_algebraic = None
            y0 = np.array([1])

        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(ExponentialDecayModel(), t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        self.assertEqual(solution.termination, "final time")

        # Exponential decay with input
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="cvodes")

        exponential_decay = -u * y

        class ExponentialDecayModelWithInputs:
            casadi_rhs = casadi.Function("rhs", [t, y, u], [exponential_decay])
            casadi_algebraic = None
            y0 = np.array([1])

        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(
            ExponentialDecayModelWithInputs(), t_eval, inputs={"u": 0.1}
        )
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        self.assertEqual(solution.termination, "final time")

    def test_integrate_failure(self):
        t = casadi.MX.sym("t")
        y = casadi.MX.sym("y")
        u = casadi.MX.sym("u")
        sqrt_decay = -np.sqrt(y)

        t_eval = np.linspace(0, 3, 100)
        solver = pybamm.CasadiSolver(regularity_check=False)

        class SqrtDecayModel:
            casadi_rhs = casadi.Function("rhs", [t, y, u], [sqrt_decay])
            casadi_algebraic = None
            y0 = np.array([1])

        # Expect solver to fail when y goes negative
        with self.assertRaises(pybamm.SolverError):
            solver.integrate(SqrtDecayModel, t_eval)

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
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
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
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.CasadiSolver(mode="fast", rtol=1e-8, atol=1e-8, method="idas")
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
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
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

    def test_model_solver_with_inputs(self):
        # Create model
        model = pybamm.BaseModel()
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
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval, inputs={"rate": 0.1})
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_array_equal(solution.t, t_eval[: len(solution.t)])
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t), rtol=1e-06)

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
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="idas")
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
