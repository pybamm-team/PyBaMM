"""Tests for the Rust-backed ``pybamm.NonlinearSolver`` root solve.

Exercises the dlsym Rust ``StandaloneNewtonSolver`` end-to-end through
``_set_up_root_solver_rust`` (``model.convert_to_format == "rust"``), the only
path that reaches the Rust Newton C++ constructor.
"""

import numpy as np
import pytest

import pybamm


class TestNonlinearSolverRust:
    def _solve_rust(self, model, t_eval, inputs=None, use_sparse=False):
        pybamm.Discretisation().process_model(model)
        model.convert_to_format = "rust"
        return pybamm.NonlinearSolver(use_sparse=use_sparse).solve(
            model, t_eval, inputs=inputs
        )

    def test_simple_root_find_rust(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + 2}
        model.initial_conditions = {var: 2}

        solution = self._solve_rust(model, np.linspace(0, 1, 10))
        np.testing.assert_allclose(solution.y, -2, atol=1e-8)

    def test_solve_with_input_rust(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + pybamm.InputParameter("param")}
        model.initial_conditions = {var: 2}

        solution = self._solve_rust(model, np.linspace(0, 1, 10), inputs={"param": 7})
        np.testing.assert_allclose(solution.y, -7, atol=1e-8)

    def test_sparse_matches_dense_rust(self):
        def build():
            model = pybamm.BaseModel()
            var = pybamm.Variable("var")
            model.algebraic = {var: var - 5}
            model.initial_conditions = {var: 1}
            return model

        t_eval = np.linspace(0, 1, 5)
        sol_sparse = self._solve_rust(build(), t_eval, use_sparse=True)
        sol_dense = self._solve_rust(build(), t_eval, use_sparse=False)
        np.testing.assert_allclose(sol_sparse.y, sol_dense.y, atol=1e-10)

    def test_compile_option_is_rejected(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + 2}
        model.initial_conditions = {var: 2}
        pybamm.Discretisation().process_model(model)
        model.convert_to_format = "rust"

        solver = pybamm.NonlinearSolver(options={"compile": True})
        with pytest.raises(pybamm.SolverError, match=r"CasADi-only"):
            solver.solve(model, np.linspace(0, 1, 5))

    def test_sensitivity_extended_states_are_rejected(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var + pybamm.InputParameter("param")}
        model.initial_conditions = {var: 2}
        pybamm.Discretisation().process_model(model)
        model.convert_to_format = "rust"
        # A y0 longer than len_rhs_and_alg is how a sensitivity-extended
        # state block reaches the root solver.
        model.y0_list = [np.zeros(model.len_rhs_and_alg + 1)]

        solver = pybamm.NonlinearSolver()
        with pytest.raises(pybamm.SolverError, match=r"sensitivity-extended"):
            solver._set_up_root_solver_rust(model, {"param": 7.0})

    def test_matches_casadi_newton(self):
        """The Rust and CasADi Newton drivers must agree on the same system."""

        def build():
            model = pybamm.BaseModel()
            var1 = pybamm.Variable("var1")
            var2 = pybamm.Variable("var2")
            model.algebraic = {var1: var1 - 3, var2: 2 * var1 - var2}
            model.initial_conditions = {
                var1: pybamm.Scalar(1),
                var2: pybamm.Scalar(4),
            }
            return model

        t_eval = np.linspace(0, 1, 5)
        sol_rust = self._solve_rust(build(), t_eval)

        model_casadi = build()
        pybamm.Discretisation().process_model(model_casadi)
        sol_casadi = pybamm.NonlinearSolver().solve(model_casadi, t_eval)

        np.testing.assert_allclose(sol_rust.y, sol_casadi.y, rtol=1e-7, atol=1e-7)
