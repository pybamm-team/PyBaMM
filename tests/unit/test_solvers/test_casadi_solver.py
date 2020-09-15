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

        # Without grid
        solver = pybamm.CasadiSolver(mode="safe without grid", rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval, inputs={"rate": 0.1})
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t), rtol=1e-04)
        solution = solver.solve(model, t_eval, inputs={"rate": 1.1})
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_allclose(solution.y[0], np.exp(-1.1 * solution.t), rtol=1e-04)

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
        for idx, t in enumerate(t_eval):
            np.testing.assert_array_almost_equal(
                sens[40 * idx : 40 * (idx + 1), :],
                -2 * t * np.exp(-p * t) * np.eye(40),
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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
