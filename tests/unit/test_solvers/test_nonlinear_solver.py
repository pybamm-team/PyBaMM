"""Tests for pybamm.NonlinearSolver (C++ Newton algebraic solver wrapper)."""

import gc
import weakref

import numpy as np
import pytest

import pybamm


class TestNonlinearSolver:
    def test_nonlinear_solver_init(self):
        solver = pybamm.NonlinearSolver(
            tol=1e-8, step_tol=1e-5, max_iter=200, max_backtracks=10,
            use_sparse=False,
        )
        assert solver.tol == 1e-8
        assert solver.step_tol == 1e-5
        assert solver.max_iter == 200
        assert solver.max_backtracks == 10
        assert solver.use_sparse is False
        assert solver._algebraic_solver is True

        solver.tol = 1e-9
        assert solver.tol == 1e-9
        solver.step_tol = 1e-6
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
        np.testing.assert_array_almost_equal(solution.y, -2, decimal=8)

    def test_solve_with_input(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + pybamm.InputParameter("param")}
        model.initial_conditions = {var: 2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.NonlinearSolver()
        solution = solver.solve(
            model, np.linspace(0, 1, 10), inputs={"param": 7},
        )
        np.testing.assert_array_almost_equal(solution.y, -7, decimal=8)

    def test_model_solver_with_time(self):
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
        np.testing.assert_allclose(
            solution["var1"].data.flatten(), sol[0, :], rtol=1e-7, atol=1e-6,
        )
        np.testing.assert_allclose(
            solution["var2"].data.flatten(), sol[1, :], rtol=1e-7, atol=1e-6,
        )

    def test_model_solver_with_time_not_changing(self):
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

    def test_solver_caching(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + 2}
        model.initial_conditions = {var: 2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.NonlinearSolver()
        solver.solve(model, [0])
        assert hasattr(model, "algebraic_root_solver")

        cached_solver = model.algebraic_root_solver
        solver.solve(model, [0])
        assert model.algebraic_root_solver is cached_solver

    def test_sparse_and_dense_modes(self):
        model_sparse = pybamm.BaseModel()
        var_s = pybamm.Variable("var")
        model_sparse.algebraic = {var_s: var_s - 5}
        model_sparse.initial_conditions = {var_s: 1}
        pybamm.Discretisation().process_model(model_sparse)

        model_dense = pybamm.BaseModel()
        var_d = pybamm.Variable("var")
        model_dense.algebraic = {var_d: var_d - 5}
        model_dense.initial_conditions = {var_d: 1}
        pybamm.Discretisation().process_model(model_dense)

        t_eval = np.linspace(0, 1, 5)
        sol_s = pybamm.NonlinearSolver(use_sparse=True).solve(model_sparse, t_eval)
        sol_d = pybamm.NonlinearSolver(use_sparse=False).solve(model_dense, t_eval)

        np.testing.assert_allclose(sol_s.y, sol_d.y, atol=1e-10)

    def test_matches_casadi_algebraic_solver(self):
        from pybamm.models.full_battery_models.lithium_ion.electrode_soh import (
            _ElectrodeSOH,
        )

        li_param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Chen2020")

        esoh_model = _ElectrodeSOH(param=li_param)
        parameter_values.process_model(esoh_model)

        Q_n = parameter_values.evaluate(li_param.n.Q_init)
        Q_p = parameter_values.evaluate(li_param.p.Q_init)
        Q_Li = parameter_values.evaluate(li_param.Q_Li_particles_init)
        inputs = {"Q_Li": Q_Li, "Q_n": Q_n, "Q_p": Q_p}

        sol_newton = pybamm.NonlinearSolver(tol=1e-10, step_tol=0).solve(
            esoh_model, [0], inputs=inputs,
        )

        model2 = _ElectrodeSOH(param=li_param)
        parameter_values.process_model(model2)
        sol_casadi = pybamm.CasadiAlgebraicSolver(tol=1e-10).solve(
            model2, [0], inputs=inputs,
        )

        np.testing.assert_allclose(sol_newton.y, sol_casadi.y, rtol=1e-3, atol=1e-5)

    def test_used_in_composite_solver(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.algebraic = {var: var - 7}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        model.variables = {"var": var}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        composite = pybamm.CompositeSolver([
            pybamm.NonlinearSolver(),
            pybamm.CasadiAlgebraicSolver(),
        ])
        solution = composite.solve(model)
        np.testing.assert_allclose(solution.y, 7, rtol=1e-6)

    def test_convergence_failure_raises(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var ** 2 + 1}
        model.initial_conditions = {var: 2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.NonlinearSolver(max_iter=50)
        with pytest.raises(pybamm.SolverError, match="Could not find acceptable"):
            solver.solve(model, [0])
