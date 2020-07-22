#
# Tests for the Casadi Solver class
#
import pybamm
import unittest
import numpy as np
from tests import get_mesh_for_testing, get_discretisation_for_testing
from scipy.sparse import eye
from scipy.optimize import least_squares


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
        model_disc = disc.process_model(model, inplace=False)
        # Solve
        solver = pybamm.CasadiSolver(mode="fast", rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model_disc, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_array_almost_equal(
            solution.y[0], np.exp(0.1 * solution.t), decimal=5
        )

        # Safe mode (enforce events that won't be triggered)
        model.events = [pybamm.Event("an event", var + 1)]
        disc.process_model(model)
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_array_almost_equal(
            solution.y[0], np.exp(0.1 * solution.t), decimal=5
        )

        # Safe mode, without grid (enforce events that won't be triggered)
        solver = pybamm.CasadiSolver(mode="safe without grid", rtol=1e-8, atol=1e-8)
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
        model_disc = disc.process_model(model, inplace=False)

        solver = pybamm.CasadiSolver(extra_options_call={"regularity_check": False})
        # Solve with failure at t=2
        t_eval = np.linspace(0, 20, 100)
        with self.assertRaises(pybamm.SolverError):
            solver.solve(model_disc, t_eval)
        # Solve with failure at t=0
        model.initial_conditions = {var: 0}
        model_disc = disc.process_model(model, inplace=False)
        t_eval = np.linspace(0, 20, 100)
        with self.assertRaises(pybamm.SolverError):
            solver.solve(model_disc, t_eval)

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

        # Solve using "safe" mode
        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y[0], 1.5)
        np.testing.assert_array_less(solution.y[-1], 2.5 + 1e-10)
        np.testing.assert_array_almost_equal(
            solution.y[0], np.exp(0.1 * solution.t), decimal=5
        )
        np.testing.assert_array_almost_equal(
            solution.y[-1], 2 * np.exp(0.1 * solution.t), decimal=5
        )

        # Solve using "safe" mode with debug off
        pybamm.settings.debug_mode = False
        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-8, atol=1e-8, dt_max=1)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y[0], 1.5)
        np.testing.assert_array_less(solution.y[-1], 2.5 + 1e-10)
        # test the last entry is exactly 2.5
        np.testing.assert_array_almost_equal(solution.y[-1, -1], 2.5, decimal=2)
        np.testing.assert_array_almost_equal(
            solution.y[0], np.exp(0.1 * solution.t), decimal=5
        )
        np.testing.assert_array_almost_equal(
            solution.y[-1], 2 * np.exp(0.1 * solution.t), decimal=5
        )
        pybamm.settings.debug_mode = True

        # Try dt_max=0 to enforce using all timesteps
        solver = pybamm.CasadiSolver(dt_max=0, rtol=1e-8, atol=1e-8)
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

        # Test when an event returns nan
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        model.events = [
            pybamm.Event("event", var - 1.02),
            pybamm.Event("sqrt event", pybamm.sqrt(1.0199 - var)),
        ]
        disc = pybamm.Discretisation()
        disc.process_model(model)
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y[0], 1.02 + 1e-10)
        np.testing.assert_array_almost_equal(solution.y[0, -1], 1.02, decimal=2)

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
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t), rtol=1e-04)

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


class TestCasadiSolverSensitivity(unittest.TestCase):
    def test_solve_with_symbolic_input(self):
        # Simple system: a single differential equation
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.rhs = {var: pybamm.InputParameter("param")}
        model.initial_conditions = {var: 2}
        model.variables = {"var": var}

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiSolver()
        t_eval = np.linspace(0, 1)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_almost_equal(
            solution["var"].value({"param": 7}).full().flatten(), 2 + 7 * t_eval
        )
        np.testing.assert_array_almost_equal(
            solution["var"].value({"param": -3}).full().flatten(), 2 - 3 * t_eval
        )
        np.testing.assert_array_almost_equal(
            solution["var"].sensitivity({"param": 3}).full().flatten(), t_eval
        )

    def test_least_squares_fit(self):
        # Simple system: a single algebraic equation
        var1 = pybamm.Variable("var1", domain="negative electrode")
        var2 = pybamm.Variable("var2", domain="negative electrode")
        model = pybamm.BaseModel()
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        model.rhs = {var1: -var1}
        model.algebraic = {var2: (var2 - p)}
        model.initial_conditions = {var1: 1, var2: 3}
        model.variables = {"objective": (var2 - q) ** 2 + (p - 3) ** 2}

        # create discretisation
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiSolver()
        solution = solver.solve(model, np.linspace(0, 1))
        sol_var = solution["objective"]

        def objective(x):
            return sol_var.value({"p": x[0], "q": x[1]}).full().flatten()

        # without jacobian
        lsq_sol = least_squares(objective, [2, 2], method="lm")
        np.testing.assert_array_almost_equal(lsq_sol.x, [3, 3], decimal=3)

        def jac(x):
            return sol_var.sensitivity({"p": x[0], "q": x[1]})

        # with jacobian
        lsq_sol = least_squares(objective, [2, 2], jac=jac, method="lm")
        np.testing.assert_array_almost_equal(lsq_sol.x, [3, 3], decimal=3)

    def test_solve_with_symbolic_input_1D_scalar_input(self):
        var = pybamm.Variable("var", "negative electrode")
        model = pybamm.BaseModel()
        param = pybamm.InputParameter("param")
        model.rhs = {var: -param * var}
        model.initial_conditions = {var: 2}
        model.variables = {"var": var}

        # create discretisation
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve - scalar input
        solver = pybamm.CasadiSolver()
        t_eval = np.linspace(0, 1)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_almost_equal(
            solution["var"].value({"param": 7}),
            np.repeat(2 * np.exp(-7 * t_eval), 40)[:, np.newaxis],
            decimal=4,
        )
        np.testing.assert_array_almost_equal(
            solution["var"].value({"param": 3}),
            np.repeat(2 * np.exp(-3 * t_eval), 40)[:, np.newaxis],
            decimal=4,
        )
        np.testing.assert_array_almost_equal(
            solution["var"].sensitivity({"param": 3}),
            np.repeat(
                -2 * t_eval * np.exp(-3 * t_eval), disc.mesh["negative electrode"].npts
            )[:, np.newaxis],
            decimal=4,
        )

    def test_solve_with_symbolic_input_1D_vector_input(self):
        var = pybamm.Variable("var", "negative electrode")
        model = pybamm.BaseModel()
        param = pybamm.InputParameter("param", "negative electrode")
        model.rhs = {var: -param * var}
        model.initial_conditions = {var: 2}
        model.variables = {"var": var}

        # create discretisation
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve - scalar input
        solver = pybamm.CasadiSolver()
        solution = solver.solve(model, np.linspace(0, 1))
        n = disc.mesh["negative electrode"].npts

        solver = pybamm.CasadiSolver()
        t_eval = np.linspace(0, 1)
        solution = solver.solve(model, t_eval)
        p = np.linspace(0, 1, n)[:, np.newaxis]
        np.testing.assert_array_almost_equal(
            solution["var"].value({"param": 3 * np.ones(n)}),
            np.repeat(2 * np.exp(-3 * t_eval), 40)[:, np.newaxis],
            decimal=4,
        )
        np.testing.assert_array_almost_equal(
            solution["var"].value({"param": 2 * p}),
            2 * np.exp(-2 * p * t_eval).T.reshape(-1, 1),
            decimal=4,
        )
        np.testing.assert_array_almost_equal(
            solution["var"].sensitivity({"param": 3 * np.ones(n)}),
            np.kron(-2 * t_eval * np.exp(-3 * t_eval), np.eye(40)).T,
            decimal=4,
        )

        sens = solution["var"].sensitivity({"param": p}).full()
        for idx in range(len(t_eval)):
            np.testing.assert_array_almost_equal(
                sens[40 * idx : 40 * (idx + 1), :],
                -2 * t_eval[idx] * np.exp(-p * t_eval[idx]) * np.eye(40),
                decimal=4,
            )

    def test_solve_with_symbolic_input_in_initial_conditions(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.rhs = {var: -var}
        model.initial_conditions = {var: pybamm.InputParameter("param")}
        model.variables = {"var": var}

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiSolver(atol=1e-10, rtol=1e-10)
        t_eval = np.linspace(0, 1)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_almost_equal(
            solution["var"].value({"param": 7}), 7 * np.exp(-t_eval)[np.newaxis, :]
        )
        np.testing.assert_array_almost_equal(
            solution["var"].value({"param": 3}), 3 * np.exp(-t_eval)[np.newaxis, :]
        )
        np.testing.assert_array_almost_equal(
            solution["var"].sensitivity({"param": 3}), np.exp(-t_eval)[:, np.newaxis]
        )

    def test_least_squares_fit_input_in_initial_conditions(self):
        # Simple system: a single algebraic equation
        var1 = pybamm.Variable("var1", domain="negative electrode")
        var2 = pybamm.Variable("var2", domain="negative electrode")
        model = pybamm.BaseModel()
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        model.rhs = {var1: -var1}
        model.algebraic = {var2: (var2 - p)}
        model.initial_conditions = {var1: 1, var2: p}
        model.variables = {"objective": (var2 - q) ** 2 + (p - 3) ** 2}

        # create discretisation
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiSolver()
        solution = solver.solve(model, np.linspace(0, 1))
        sol_var = solution["objective"]

        def objective(x):
            return sol_var.value({"p": x[0], "q": x[1]}).full().flatten()

        # without jacobian
        lsq_sol = least_squares(objective, [2, 2], method="lm")
        np.testing.assert_array_almost_equal(lsq_sol.x, [3, 3], decimal=3)


class TestCasadiSolverODEsWithForwardSensitivityEquations(unittest.TestCase):
    def test_solve_sensitivity_scalar_var_scalar_input(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        p = pybamm.InputParameter("p")
        model.rhs = {var: p * var}
        model.initial_conditions = {var: 1}
        model.variables = {"var squared": var ** 2}

        # Solve
        # Make sure that passing in extra options works
        solver = pybamm.CasadiSolver(
            mode="fast", rtol=1e-10, atol=1e-10, sensitivity="explicit forward"
        )
        t_eval = np.linspace(0, 1, 80)
        solution = solver.solve(model, t_eval, inputs={"p": 0.1})
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(
            solution.sensitivity["p"],
            (solution.t * np.exp(0.1 * solution.t))[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var squared"].data, np.exp(0.1 * solution.t) ** 2
        )
        np.testing.assert_allclose(
            solution["var squared"].sensitivity["p"],
            (2 * np.exp(0.1 * solution.t) * solution.t * np.exp(0.1 * solution.t))[
                :, np.newaxis
            ],
        )

        # More complicated model
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        r = pybamm.InputParameter("r")
        s = pybamm.InputParameter("s")
        model.rhs = {var: p * q}
        model.initial_conditions = {var: r}
        model.variables = {"var times s": var * s}

        # Solve
        # Make sure that passing in extra options works
        solver = pybamm.CasadiSolver(
            rtol=1e-10, atol=1e-10, sensitivity="explicit forward"
        )
        t_eval = np.linspace(0, 1, 80)
        solution = solver.solve(
            model, t_eval, inputs={"p": 0.1, "q": 2, "r": -1, "s": 0.5}
        )
        np.testing.assert_allclose(solution.y[0], -1 + 0.2 * solution.t)
        np.testing.assert_allclose(
            solution.sensitivity["p"], (2 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution.sensitivity["q"], (0.1 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(solution.sensitivity["r"], 1)
        np.testing.assert_allclose(solution.sensitivity["s"], 0)
        np.testing.assert_allclose(
            solution.sensitivity["all"],
            np.hstack(
                [
                    solution.sensitivity["p"],
                    solution.sensitivity["q"],
                    solution.sensitivity["r"],
                    solution.sensitivity["s"],
                ]
            ),
        )
        np.testing.assert_allclose(
            solution["var times s"].data, 0.5 * (-1 + 0.2 * solution.t)
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivity["p"],
            0.5 * (2 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivity["q"],
            0.5 * (0.1 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(solution["var times s"].sensitivity["r"], 0.5)
        np.testing.assert_allclose(
            solution["var times s"].sensitivity["s"],
            (-1 + 0.2 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivity["all"],
            np.hstack(
                [
                    solution["var times s"].sensitivity["p"],
                    solution["var times s"].sensitivity["q"],
                    solution["var times s"].sensitivity["r"],
                    solution["var times s"].sensitivity["s"],
                ]
            ),
        )

    def test_solve_sensitivity_vector_var_scalar_input(self):
        var = pybamm.Variable("var", "negative electrode")
        model = pybamm.BaseModel()
        # Set length scales to avoid warning
        model.length_scales = {"negative electrode": 1}
        param = pybamm.InputParameter("param")
        model.rhs = {var: -param * var}
        model.initial_conditions = {var: 2}
        model.variables = {"var": var}

        # create discretisation
        disc = get_discretisation_for_testing()
        disc.process_model(model)
        n = disc.mesh["negative electrode"].npts

        # Solve - scalar input
        solver = pybamm.CasadiSolver(sensitivity="explicit forward")
        t_eval = np.linspace(0, 1)
        solution = solver.solve(model, t_eval, inputs={"param": 7})
        np.testing.assert_array_almost_equal(
            solution["var"].data, np.tile(2 * np.exp(-7 * t_eval), (n, 1)), decimal=4,
        )
        np.testing.assert_array_almost_equal(
            solution["var"].sensitivity["param"],
            np.repeat(-2 * t_eval * np.exp(-7 * t_eval), n)[:, np.newaxis],
            decimal=4,
        )

        # More complicated model
        # Create model
        model = pybamm.BaseModel()
        # Set length scales to avoid warning
        model.length_scales = {"negative electrode": 1}
        var = pybamm.Variable("var", "negative electrode")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        r = pybamm.InputParameter("r")
        s = pybamm.InputParameter("s")
        model.rhs = {var: p * q}
        model.initial_conditions = {var: r}
        model.variables = {"var times s": var * s}

        # Discretise
        disc.process_model(model)

        # Solve
        # Make sure that passing in extra options works
        solver = pybamm.CasadiSolver(
            rtol=1e-10, atol=1e-10, sensitivity="explicit forward"
        )
        t_eval = np.linspace(0, 1, 80)
        solution = solver.solve(
            model, t_eval, inputs={"p": 0.1, "q": 2, "r": -1, "s": 0.5}
        )
        np.testing.assert_allclose(solution.y, np.tile(-1 + 0.2 * solution.t, (n, 1)))
        np.testing.assert_allclose(
            solution.sensitivity["p"], np.repeat(2 * solution.t, n)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution.sensitivity["q"], np.repeat(0.1 * solution.t, n)[:, np.newaxis],
        )
        np.testing.assert_allclose(solution.sensitivity["r"], 1)
        np.testing.assert_allclose(solution.sensitivity["s"], 0)
        np.testing.assert_allclose(
            solution.sensitivity["all"],
            np.hstack(
                [
                    solution.sensitivity["p"],
                    solution.sensitivity["q"],
                    solution.sensitivity["r"],
                    solution.sensitivity["s"],
                ]
            ),
        )
        np.testing.assert_allclose(
            solution["var times s"].data, np.tile(0.5 * (-1 + 0.2 * solution.t), (n, 1))
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivity["p"],
            np.repeat(0.5 * (2 * solution.t), n)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivity["q"],
            np.repeat(0.5 * (0.1 * solution.t), n)[:, np.newaxis],
        )
        np.testing.assert_allclose(solution["var times s"].sensitivity["r"], 0.5)
        np.testing.assert_allclose(
            solution["var times s"].sensitivity["s"],
            np.repeat(-1 + 0.2 * solution.t, n)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivity["all"],
            np.hstack(
                [
                    solution["var times s"].sensitivity["p"],
                    solution["var times s"].sensitivity["q"],
                    solution["var times s"].sensitivity["r"],
                    solution["var times s"].sensitivity["s"],
                ]
            ),
        )

    def test_solve_sensitivity_scalar_var_vector_input(self):
        var = pybamm.Variable("var", "negative electrode")
        model = pybamm.BaseModel()
        # Set length scales to avoid warning
        model.length_scales = {"negative electrode": 1}

        param = pybamm.InputParameter("param", "negative electrode")
        model.rhs = {var: -param * var}
        model.initial_conditions = {var: 2}
        model.variables = {
            "var": var,
            "integral of var": pybamm.Integral(var, pybamm.standard_spatial_vars.x_n),
        }

        # create discretisation
        mesh = get_mesh_for_testing(xpts=5)
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        n = disc.mesh["negative electrode"].npts

        # Solve - constant input
        solver = pybamm.CasadiSolver(
            mode="fast", rtol=1e-10, atol=1e-10, sensitivity="explicit forward"
        )
        t_eval = np.linspace(0, 1)
        solution = solver.solve(model, t_eval, inputs={"param": 7 * np.ones(n)})
        l_n = mesh["negative electrode"].edges[-1]
        np.testing.assert_array_almost_equal(
            solution["var"].data, np.tile(2 * np.exp(-7 * t_eval), (n, 1)), decimal=4,
        )

        np.testing.assert_array_almost_equal(
            solution["var"].sensitivity["param"],
            np.vstack([np.eye(n) * -2 * t * np.exp(-7 * t) for t in t_eval]),
        )
        np.testing.assert_array_almost_equal(
            solution["integral of var"].data, 2 * np.exp(-7 * t_eval) * l_n, decimal=4,
        )
        np.testing.assert_array_almost_equal(
            solution["integral of var"].sensitivity["param"],
            np.tile(-2 * t_eval * np.exp(-7 * t_eval) * l_n / n, (n, 1)).T,
        )

        # Solve - linspace input
        p_eval = np.linspace(1, 2, n)
        solution = solver.solve(model, t_eval, inputs={"param": p_eval})
        l_n = mesh["negative electrode"].edges[-1]
        np.testing.assert_array_almost_equal(
            solution["var"].data, 2 * np.exp(-p_eval[:, np.newaxis] * t_eval), decimal=4
        )
        np.testing.assert_array_almost_equal(
            solution["var"].sensitivity["param"],
            np.vstack([np.diag(-2 * t * np.exp(-p_eval * t)) for t in t_eval]),
        )

        np.testing.assert_array_almost_equal(
            solution["integral of var"].data,
            np.sum(
                2
                * np.exp(-p_eval[:, np.newaxis] * t_eval)
                * mesh["negative electrode"].d_edges[:, np.newaxis],
                axis=0,
            ),
        )
        np.testing.assert_array_almost_equal(
            solution["integral of var"].sensitivity["param"],
            np.vstack([-2 * t * np.exp(-p_eval * t) * l_n / n for t in t_eval]),
        )


class TestCasadiSolverDAEsWithForwardSensitivityEquations(unittest.TestCase):
    def test_solve_sensitivity_scalar_var_scalar_input(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        var2 = pybamm.Variable("var2")
        p = pybamm.InputParameter("p")
        model.rhs = {var: p * var}
        model.algebraic = {var2: var2 - p}
        model.initial_conditions = {var: 1, var2: p}
        model.variables = {"prod": var * var2}

        # Solve
        # Make sure that passing in extra options works
        solver = pybamm.CasadiSolver(
            mode="fast", rtol=1e-10, atol=1e-10, sensitivity="explicit forward"
        )
        t_eval = np.linspace(0, 1, 80)
        solution = solver.solve(model, t_eval, inputs={"p": 0.1})
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[1], 0.1)
        np.testing.assert_allclose(
            solution.sensitivity["p"],
            np.hstack(
                [
                    (solution.t * np.exp(0.1 * solution.t))[:, np.newaxis],
                    np.ones((len(t_eval), 1)),
                ]
            ).reshape(2 * len(t_eval), 1),
        )
        np.testing.assert_allclose(
            solution["prod"].data, 0.1 * np.exp(0.1 * solution.t)
        )
        np.testing.assert_allclose(
            solution["prod"].sensitivity["p"],
            ((1 + 0.1 * solution.t) * np.exp(0.1 * solution.t))[:, np.newaxis],
        )

        # More complicated model
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        var2 = pybamm.Variable("var2")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        r = pybamm.InputParameter("r")
        s = pybamm.InputParameter("s")
        model.rhs = {var: p}
        model.algebraic = {var2: var2 - q}
        model.initial_conditions = {var: r, var2: q}
        model.variables = {"var prod times s": var * var2 * s}

        # Solve
        # Make sure that passing in extra options works
        solver = pybamm.CasadiSolver(
            rtol=1e-10, atol=1e-10, sensitivity="explicit forward"
        )
        t_eval = np.linspace(0, 1, 3)
        solution = solver.solve(
            model, t_eval, inputs={"p": 0.1, "q": 2, "r": -1, "s": 0.5}
        )
        np.testing.assert_allclose(solution.y[0], -1 + 0.1 * solution.t)
        np.testing.assert_allclose(solution.y[1], 2)
        n_t = len(t_eval)
        zeros = np.zeros((n_t, 1))
        ones = np.ones((n_t, 1))
        np.testing.assert_allclose(
            solution.sensitivity["p"],
            np.hstack([solution.t[:, np.newaxis], zeros]).reshape(2 * n_t, 1),
        )
        np.testing.assert_allclose(
            solution.sensitivity["q"], np.hstack([zeros, ones]).reshape(2 * n_t, 1),
        )
        np.testing.assert_allclose(
            solution.sensitivity["r"], np.hstack([ones, zeros]).reshape(2 * n_t, 1)
        )
        np.testing.assert_allclose(solution.sensitivity["s"], 0)
        np.testing.assert_allclose(
            solution.sensitivity["all"],
            np.hstack(
                [
                    solution.sensitivity["p"],
                    solution.sensitivity["q"],
                    solution.sensitivity["r"],
                    solution.sensitivity["s"],
                ]
            ),
        )
        np.testing.assert_allclose(
            solution["var prod times s"].data, 0.5 * 2 * (-1 + 0.1 * solution.t)
        )
        np.testing.assert_allclose(
            solution["var prod times s"].sensitivity["p"],
            0.5 * (2 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var prod times s"].sensitivity["q"],
            0.5 * (-1 + 0.1 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(solution["var prod times s"].sensitivity["r"], 1)
        np.testing.assert_allclose(
            solution["var prod times s"].sensitivity["s"],
            2 * (-1 + 0.1 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var prod times s"].sensitivity["all"],
            np.hstack(
                [
                    solution["var prod times s"].sensitivity["p"],
                    solution["var prod times s"].sensitivity["q"],
                    solution["var prod times s"].sensitivity["r"],
                    solution["var prod times s"].sensitivity["s"],
                ]
            ),
        )

    def test_solve_sensitivity_vector_var_scalar_input(self):
        var = pybamm.Variable("var", "negative electrode")
        var2 = pybamm.Variable("var2", "negative electrode")
        model = pybamm.BaseModel()
        # Set length scales to avoid warning
        model.length_scales = {"negative electrode": 1}
        param = pybamm.InputParameter("param")
        model.rhs = {var: -param * var}
        model.algebraic = {var2: var2 - param}
        model.initial_conditions = {var: 2, var2: param}
        model.variables = {"prod": var * var2}

        # create discretisation
        disc = get_discretisation_for_testing()
        disc.process_model(model)
        n = disc.mesh["negative electrode"].npts

        # Solve - scalar input
        solver = pybamm.CasadiSolver(sensitivity="explicit forward")
        t_eval = np.linspace(0, 1)
        solution = solver.solve(model, t_eval, inputs={"param": 7})
        np.testing.assert_array_almost_equal(
            solution["prod"].data,
            np.tile(2 * 7 * np.exp(-7 * t_eval), (n, 1)),
            decimal=4,
        )
        np.testing.assert_array_almost_equal(
            solution["prod"].sensitivity["param"],
            np.repeat(2 * (1 - 7 * t_eval) * np.exp(-7 * t_eval), n)[:, np.newaxis],
            decimal=4,
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
