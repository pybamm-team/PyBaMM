"""Tests for the Rust-backed ``pybamm.NonlinearSolver`` root solve."""

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

    def test_klu_pivot_tolerance_reaches_newton_solver(self, monkeypatch):
        nonlinear_solver = pybamm.solvers.nonlinear_solver
        idaklu = nonlinear_solver.idaklu
        received = []

        class RecordingIdaklu:
            def __getattr__(self, name):
                return getattr(idaklu, name)

            def StandaloneNewtonSolver(self, **kwargs):
                received.append(kwargs["klu_pivot_tolerance"])
                return idaklu.StandaloneNewtonSolver(**kwargs)

        monkeypatch.setattr(nonlinear_solver, "idaklu", RecordingIdaklu())
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        model.algebraic = {var: var**3 - 8}
        model.initial_conditions = {var: 1}
        pybamm.Discretisation().process_model(model)
        model.convert_to_format = "rust"

        solver = pybamm.NonlinearSolver(
            use_sparse=True, options={"klu_pivot_tolerance": 0.001}
        )
        solution = solver.solve(model, np.linspace(0, 1, 5))

        assert received == [0.001]
        np.testing.assert_allclose(solution.y, 2, rtol=1e-6)

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
        # A sensitivity-extended state block reaches the root solver as a longer y0.
        model.y0_list = [np.zeros(model.len_rhs_and_alg + 1)]

        solver = pybamm.NonlinearSolver()
        with pytest.raises(pybamm.SolverError, match=r"sensitivity-extended"):
            solver._set_up_root_solver_rust(model, {"param": 7.0})

    @staticmethod
    def _input_model(n_vars=1):
        model = pybamm.BaseModel()
        param = pybamm.InputParameter("param")
        for i in range(n_vars):
            var = pybamm.Variable(f"var{i}")
            model.algebraic[var] = var + param
            model.initial_conditions[var] = pybamm.Scalar(2)
        pybamm.Discretisation().process_model(model)
        model.convert_to_format = "rust"
        return model

    @pytest.mark.parametrize(
        ("y0_alg", "p_vec", "match"),
        [
            ([2.0], [], r"\[y_diff; inputs\] of length 1 but got 0"),
            ([2.0], [7.0, 1.0], r"\[y_diff; inputs\] of length 1 but got 2"),
            ([2.0, 2.0], [7.0], r"y0_alg of length 1 but got 2"),
        ],
    )
    def test_buffers_of_another_length_are_rejected(self, y0_alg, p_vec, match):
        # C++ reads both at the set-up lengths, so a short one would be overrun
        setup = pybamm.NonlinearSolver()._set_up_root_solver_rust(
            self._input_model(), {"param": 7.0}
        )
        with pytest.raises(pybamm.SolverError, match=match):
            setup.solve_batch(np.zeros(1), np.array(y0_alg), np.array(p_vec))
        success, y = setup.solve_batch(np.zeros(1), np.array([2.0]), np.array([7.0]))
        assert success
        np.testing.assert_allclose(y, -7.0)

    def test_a_rust_model_of_another_size_is_rejected(self, monkeypatch):
        from pybamm.solvers.rust_lowering import RustModelLowering

        compile_model = RustModelLowering.compile
        other = RustModelLowering(self._input_model(n_vars=2), {"param": 7.0})
        other.state_residual(algebraic_only=True)
        other.algebraic_block()
        monkeypatch.setattr(
            RustModelLowering, "compile", lambda _: compile_model(other)
        )

        solver = pybamm.NonlinearSolver()
        with pytest.raises(
            pybamm.SolverError, match=r"has 2 states, 2 of them algebraic.*1 and 1"
        ):
            solver._set_up_root_solver_rust(self._input_model(), {"param": 7.0})

    def test_matches_casadi_newton(self):
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
        model_casadi.convert_to_format = "casadi"
        pybamm.Discretisation().process_model(model_casadi)
        sol_casadi = pybamm.NonlinearSolver().solve(model_casadi, t_eval)

        np.testing.assert_allclose(sol_rust.y, sol_casadi.y, rtol=1e-7, atol=1e-7)
