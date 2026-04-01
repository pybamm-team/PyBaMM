import numpy as np
import pytest
import warnings

import pybamm


class TestNonlinearSolver:
    def test_solver_init(self):
        solver = pybamm.NonlinearSolver(rtol=1e-8, atol=1e-10, step_tol=1e-7)
        assert solver.rtol == 1e-8
        assert solver.atol == 1e-10
        assert solver.step_tol == 1e-7
        assert solver.algebraic_solver is True
        assert solver.name == "Newton algebraic solver"

    def test_solver_init_defaults(self):
        solver = pybamm.NonlinearSolver()
        assert solver.rtol == 1e-6
        assert solver.atol == 1e-6
        assert solver.step_tol == 1e-6

    def test_simple_root_find(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + 2}
        model.initial_conditions = {var: 2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.NonlinearSolver()
        solution = solver.solve(model, np.linspace(0, 1, 10))
        np.testing.assert_array_almost_equal(solution.y, -2, decimal=10)

    def test_quadratic_root(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var**2 - 4}
        model.initial_conditions = {var: pybamm.Scalar(3)}
        model.variables = {"var": var}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.NonlinearSolver()
        solution = solver.solve(model, [0])
        np.testing.assert_allclose(solution["var"].entries, 2.0, rtol=1e-10)

    def test_model_with_time(self):
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.algebraic = {var1: var1 - 3 * pybamm.t, var2: 2 * var1 - var2}
        model.initial_conditions = {var1: pybamm.Scalar(1), var2: pybamm.Scalar(4)}
        model.variables = {"var1": var1, "var2": var2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 1)
        solver = pybamm.NonlinearSolver()
        solution = solver.solve(model, t_eval)

        sol = np.vstack((3 * t_eval, 6 * t_eval))
        np.testing.assert_allclose(solution.y, sol, rtol=1e-7, atol=1e-6)

    def test_model_without_time_dependence(self):
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.algebraic = {var1: var1 - 3, var2: 2 * var1 - var2}
        model.initial_conditions = {var1: pybamm.Scalar(1), var2: pybamm.Scalar(4)}
        model.variables = {"var1": var1, "var2": var2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 1)
        solver = pybamm.NonlinearSolver()
        solution = solver.solve(model, t_eval)

        sol = np.vstack((3 + 0 * t_eval, 6 + 0 * t_eval))
        np.testing.assert_allclose(solution.y, sol, rtol=1e-7, atol=1e-6)

    def test_solve_with_input(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + pybamm.InputParameter("param")}
        model.initial_conditions = {var: 2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.NonlinearSolver()
        solution = solver.solve(model, np.linspace(0, 1, 10), inputs={"param": 7})
        np.testing.assert_array_almost_equal(solution.y, -7, decimal=10)

    def test_on_failure_raise(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var**2 + 1}
        model.initial_conditions = {var: pybamm.Scalar(1)}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.NonlinearSolver(on_failure="raise", options={"max_iterations": 10})
        with pytest.raises(pybamm.SolverError, match="Could not find acceptable solution"):
            solver.solve(model, [0])

    def test_on_failure_warn(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var**2 + 1}
        model.initial_conditions = {var: pybamm.Scalar(1)}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.NonlinearSolver(on_failure="warn", options={"max_iterations": 10})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.solve(model, [0])
            assert len(w) >= 1
            assert "Could not find acceptable solution" in str(w[0].message)

    def test_on_failure_ignore(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var**2 + 1}
        model.initial_conditions = {var: pybamm.Scalar(1)}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.NonlinearSolver(on_failure="ignore", options={"max_iterations": 10})
        sol = solver.solve(model, [0])
        assert sol is not None

    def test_correctness_vs_casadi(self):
        """Compare NonlinearSolver results against CasadiAlgebraicSolver."""
        model_newton = pybamm.BaseModel()
        model_casadi = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        alg = {var1: var1 - 3 * pybamm.t, var2: 2 * var1 - var2}
        ics = {var1: pybamm.Scalar(1), var2: pybamm.Scalar(4)}
        model_newton.algebraic = alg
        model_newton.initial_conditions = ics
        model_newton.variables = {"var1": var1, "var2": var2}
        model_casadi.algebraic = alg
        model_casadi.initial_conditions = ics
        model_casadi.variables = {"var1": var1, "var2": var2}

        disc_n = pybamm.Discretisation()
        disc_n.process_model(model_newton)
        disc_c = pybamm.Discretisation()
        disc_c.process_model(model_casadi)

        t_eval = np.linspace(0, 1, 20)

        sol_newton = pybamm.NonlinearSolver().solve(model_newton, t_eval)
        sol_casadi = pybamm.CasadiAlgebraicSolver().solve(model_casadi, t_eval)

        np.testing.assert_allclose(sol_newton.y, sol_casadi.y, rtol=1e-7, atol=1e-10)

    def test_as_root_method_idaklu(self):
        """Test NonlinearSolver as root_method for IDAKLUSolver."""
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.IDAKLUSolver(root_method="newton")
        sim = pybamm.Simulation(model, solver=solver)
        sol = sim.solve([0, 3600])

        model2 = pybamm.lithium_ion.SPM()
        solver2 = pybamm.IDAKLUSolver(root_method="casadi")
        sim2 = pybamm.Simulation(model2, solver=solver2)
        sol2 = sim2.solve([0, 3600])

        np.testing.assert_allclose(
            sol["Terminal voltage [V]"].entries,
            sol2["Terminal voltage [V]"].entries,
            rtol=1e-4,
        )

    def test_as_root_method_casadi_solver(self):
        """Test NonlinearSolver as root_method for CasadiSolver."""
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.CasadiSolver(root_method="newton")
        sim = pybamm.Simulation(model, solver=solver)
        sol = sim.solve([0, 3600])

        model2 = pybamm.lithium_ion.SPM()
        solver2 = pybamm.CasadiSolver(root_method="casadi")
        sim2 = pybamm.Simulation(model2, solver=solver2)
        sol2 = sim2.solve([0, 3600])

        np.testing.assert_allclose(
            sol["Terminal voltage [V]"].entries,
            sol2["Terminal voltage [V]"].entries,
            rtol=1e-4,
        )

    def test_options_max_iterations(self):
        """Test that max_iterations option is respected."""
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var**2 + 1}
        model.initial_conditions = {var: pybamm.Scalar(1)}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.NonlinearSolver(
            on_failure="raise", options={"max_iterations": 2}
        )
        with pytest.raises(pybamm.SolverError):
            solver.solve(model, [0])

    def test_dense_jacobian(self):
        """Test with dense Jacobian / dense linear solver."""
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + 2}
        model.initial_conditions = {var: 2}
        model.variables = {"var": var}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.NonlinearSolver(
            options={"jacobian": "dense", "linear_solver": "SUNLinSol_Dense"}
        )
        solution = solver.solve(model, [0])
        np.testing.assert_allclose(solution["var"].entries, -2, rtol=1e-10)
