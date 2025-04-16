import casadi
import pybamm
import pytest
import numpy as np


class TestCasadiAlgebraicSolver:
    def test_algebraic_solver_init(self):
        solver = pybamm.CasadiAlgebraicSolver(tol=1e-4)
        assert solver.tol == 1e-4

        solver.tol = 1e-5
        assert solver.tol == 1e-5

    def test_simple_root_find(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + 2}
        model.initial_conditions = {var: 2}

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(model, np.linspace(0, 1, 10))
        np.testing.assert_array_equal(solution.y, -2)

    def test_simple_root_find_correct_initial_guess(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + 2}
        # initial guess gives right answer
        model.initial_conditions = {var: -2}

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(model, np.linspace(0, 1, 10))
        np.testing.assert_array_equal(solution.y, -2)

    def test_root_find_fail(self):
        class Model:
            y0 = np.array([2])
            t = casadi.MX.sym("t")
            y = casadi.MX.sym("y")
            p = casadi.MX.sym("p")
            rhs = {}
            casadi_algebraic = casadi.Function("alg", [t, y, p], [y**2 + 1])
            bounds = (np.array([-np.inf]), np.array([np.inf]))
            interpolant_extrapolation_events_eval = []

            def algebraic_eval(self, t, y, inputs):
                # algebraic equation has no real root
                return y**2 + 1

        model = Model()

        solver = pybamm.CasadiAlgebraicSolver()
        with pytest.raises(
            pybamm.SolverError,
            match="Could not find acceptable solution: Error in Function",
        ):
            solver._integrate(model, np.array([0]), {})
        solver = pybamm.CasadiAlgebraicSolver(extra_options={"error_on_fail": False})
        with pytest.raises(
            pybamm.SolverError,
            match="Could not find acceptable solution: solver terminated",
        ):
            solver._integrate(model, np.array([0]), {})

        # Model returns Nan
        class NaNModel:
            y0 = np.array([-2])
            t = casadi.MX.sym("t")
            y = casadi.MX.sym("y")
            p = casadi.MX.sym("p")
            rhs = {}
            casadi_algebraic = casadi.Function("alg", [t, y, p], [y**0.5])
            bounds = (np.array([-np.inf]), np.array([np.inf]))
            interpolant_extrapolation_events_eval = []

            def algebraic_eval(self, t, y, inputs):
                # algebraic equation has no real root
                return y**0.5

        model = NaNModel()
        with pytest.raises(
            pybamm.SolverError,
            match="Could not find acceptable solution: solver returned NaNs",
        ):
            solver._integrate(model, np.array([0]), {})

    def test_model_solver_with_time(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.algebraic = {var1: var1 - 3 * pybamm.t, var2: 2 * var1 - var2}
        model.initial_conditions = {var1: pybamm.Scalar(1), var2: pybamm.Scalar(4)}
        model.variables = {"var1": var1, "var2": var2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        t_eval = np.linspace(0, 1)
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(model, t_eval)

        sol = np.vstack((3 * t_eval, 6 * t_eval))
        np.testing.assert_allclose(solution.y, sol, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(
            solution["var1"].data.flatten(), sol[0, :], rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            solution["var2"].data.flatten(), sol[1, :], rtol=1e-7, atol=1e-6
        )

    def test_model_solver_with_time_not_changing(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.algebraic = {var1: var1 - 3, var2: 2 * var1 - var2}
        model.initial_conditions = {var1: pybamm.Scalar(1), var2: pybamm.Scalar(4)}
        model.variables = {"var1": var1, "var2": var2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        t_eval = np.linspace(0, 1)
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(model, t_eval)

        sol = np.vstack((3 + 0 * t_eval, 6 + 0 * t_eval))
        np.testing.assert_allclose(solution.y, sol, rtol=1e-7, atol=1e-6)

    def test_model_solver_with_bounds(self):
        # Note: we need a better test case to test this functionality properly
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1", bounds=(0, 10))
        model.algebraic = {var1: pybamm.sin(var1) + 1}
        model.initial_conditions = {var1: pybamm.Scalar(3)}
        model.variables = {"var1": var1}

        # Solve
        solver = pybamm.CasadiAlgebraicSolver(tol=1e-12)
        solution = solver.solve(model)
        np.testing.assert_allclose(
            solution["var1"].data, 3 * np.pi / 2, rtol=1e-7, atol=1e-6
        )

    def test_solve_with_input(self):
        # Simple system: a single algebraic equation
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + pybamm.InputParameter("param")}
        model.initial_conditions = {var: 2}

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        solver = pybamm.CasadiAlgebraicSolver()
        solution = solver.solve(model, np.linspace(0, 1, 10), inputs={"param": 7})
        np.testing.assert_array_equal(solution.y, -7)
