"""Rust `CompiledModel` Jacobian parity against CasADi, for IDAKLU."""

import numpy as np
import pytest
from scipy.sparse import diags, eye

import pybamm

casadi = pytest.importorskip("casadi")

try:
    from pybamm.rust import CompiledModel, ExprGraph
except ImportError:
    pytest.skip(
        "Rust extension not available. Build with: uv sync",
        allow_module_level=True,
    )


class TestRustIDAKLUJacobianParity:
    @staticmethod
    def build_diffusion_reaction_expr(n_states: int):
        """Build ``f(y) = D * L @ y + k * exp(-y/T) * (1 - y)``, mimicking SPM/DFN."""
        sv = pybamm.StateVector(slice(0, n_states))
        t = pybamm.Time()
        D = pybamm.InputParameter("diffusivity")
        k = pybamm.InputParameter("rate_constant")
        T = pybamm.InputParameter("temperature")

        diag_data = [
            np.ones(n_states) * -2,
            np.ones(n_states - 1),
            np.ones(n_states - 1),
        ]
        laplacian = diags(diag_data, [0, -1, 1], format="csr")
        L = pybamm.Matrix(laplacian)

        diffusion = D * (L @ sv)

        reaction = (
            k
            * pybamm.exp(-sv / T)
            * (pybamm.Scalar(1.0) - sv)
            * (pybamm.Scalar(1.0) + pybamm.Scalar(0.01) * t)
        )

        return diffusion + reaction

    @staticmethod
    def build_stiff_ode_expr(n_states: int):
        """Build a stiff ODE ``f(y) = -A @ y + nonlinear_source(y)``."""
        sv = pybamm.StateVector(slice(0, n_states))
        t = pybamm.Time()
        alpha = pybamm.InputParameter("alpha")
        beta = pybamm.InputParameter("beta")

        diag_main = np.linspace(1.0, 1000.0, n_states)
        diag_off = np.ones(n_states - 1) * 0.1
        A_sparse = diags([diag_main, -diag_off, -diag_off], [0, -1, 1], format="csr")
        A = pybamm.Matrix(A_sparse)

        linear = -alpha * (A @ sv)

        nonlinear = (
            beta
            * pybamm.tanh(sv)
            * pybamm.exp(-pybamm.Scalar(0.01) * sv * sv)
            * (pybamm.Scalar(1.0) + pybamm.Scalar(0.001) * t)
        )

        return linear + nonlinear

    @staticmethod
    def _build_casadi_functions(expr, n_states: int, inputs: dict):
        """Build CasADi functions for RHS and Jacobian-vector product."""
        t_sym = casadi.MX.sym("t")
        y_sym = casadi.MX.sym("y", n_states)
        y_dot_sym = casadi.MX.sym("y_dot", 0)
        inputs_sym = {name: casadi.MX.sym(name) for name in inputs}

        casadi_symbols = {
            "t": t_sym,
            "y": y_sym,
            "y_dot": y_dot_sym,
            "inputs": inputs_sym,
        }
        f_casadi = expr.to_casadi(t_sym, y_sym, y_dot_sym, inputs_sym, casadi_symbols)

        func_inputs = [t_sym, y_sym, *inputs_sym.values()]
        f_fn = casadi.Function("f", func_inputs, [f_casadi])

        jac_expr = casadi.jacobian(f_casadi, y_sym)
        jac_fn = casadi.Function("jacobian_fn", func_inputs, [jac_expr])

        # Jacobian-vector product: J @ v - cj * M @ v (M = I for this test)
        v_sym = casadi.MX.sym("v", n_states)
        cj_sym = casadi.MX.sym("cj")
        jac_action = casadi.jtimes(f_casadi, y_sym, v_sym) - cj_sym * v_sym
        jac_inputs = [t_sym, y_sym, *inputs_sym.values(), cj_sym, v_sym]
        jac_action_fn = casadi.Function("jac_action", jac_inputs, [jac_action])

        return f_fn, jac_fn, jac_action_fn

    @staticmethod
    def _build_rust_model(expr, n_states: int):
        """Build Rust CompiledModel from PyBaMM expression."""
        graph = ExprGraph()
        rust_symbols = {}
        rust_expr = expr.to_rust(graph, rust_symbols)

        mass = eye(n_states, format="csr")
        model = CompiledModel.from_expr(
            graph,
            rust_expr,
            mass.data.astype(np.float64),
            mass.indptr.astype(np.int64),
            mass.indices.astype(np.int64),
        )
        return model

    @pytest.mark.parametrize("n_states", [10, 50, 100])
    def test_rhs_parity_diffusion_reaction(self, n_states):
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        f_fn, _, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        rng = np.random.default_rng(0)
        y = rng.standard_normal(n_states) * 0.1 + 0.5
        t = 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        casadi_result = np.array(f_fn(t, y, *inputs.values())).flatten()
        rust_result = np.array(rust_model.rhs(t, y, inputs_arr))

        np.testing.assert_allclose(rust_result, casadi_result, rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("n_states", [10, 50, 100])
    def test_rhs_parity_stiff_ode(self, n_states):
        inputs = {"alpha": 1.0, "beta": 0.5}
        expr = self.build_stiff_ode_expr(n_states)

        f_fn, _, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        rng = np.random.default_rng(0)
        y = rng.standard_normal(n_states) * 0.1 + 0.5
        t = 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        casadi_result = np.array(f_fn(t, y, *inputs.values())).flatten()
        rust_result = np.array(rust_model.rhs(t, y, inputs_arr))

        np.testing.assert_allclose(rust_result, casadi_result, rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("n_states", [10, 50, 100])
    def test_jac_mul_parity(self, n_states):
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        _, _, jac_action_fn = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        rng = np.random.default_rng(0)
        y = rng.standard_normal(n_states) * 0.1 + 0.5
        v = rng.standard_normal(n_states)
        t = 0.5
        cj = 1.0
        inputs_arr = np.array([inputs[name] for name in inputs])

        casadi_result = np.array(jac_action_fn(t, y, *inputs.values(), cj, v)).flatten()
        J = rust_model.jacobian(t, y, inputs_arr)
        M = eye(n_states, format="csr")
        rust_result = np.array(J @ v - cj * (M @ v))

        np.testing.assert_allclose(rust_result, casadi_result, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("n_states", [10, 50])
    def test_assembled_jacobian_parity(self, n_states):
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        _, jac_fn, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        rng = np.random.default_rng(0)
        y = rng.standard_normal(n_states) * 0.1 + 0.5
        t = 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        casadi_jac = np.array(jac_fn(t, y, *inputs.values()))

        # model.jacobian is the pure df/dy.
        J = rust_model.jacobian(t, y, inputs_arr)
        rust_jac = J.toarray()

        np.testing.assert_allclose(rust_jac, casadi_jac, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("n_states", [10, 50])
    def test_assembled_jacobian_with_cj(self, n_states):
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        _, jac_fn, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        rng = np.random.default_rng(0)
        y = rng.standard_normal(n_states) * 0.1 + 0.5
        t = 0.5
        cj = 1.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        casadi_jac = np.array(jac_fn(t, y, *inputs.values()))
        expected_jac = casadi_jac - cj * np.eye(n_states)

        J = rust_model.jacobian(t, y, inputs_arr)
        M = eye(n_states, format="csr")
        rust_jac = (J - cj * M).toarray()

        np.testing.assert_allclose(rust_jac, expected_jac, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("n_states", [10, 50])
    def test_jacobian_jvp_consistency(self, n_states):
        """Matrix-free tangent JVP agrees with the coloured-assembly Jacobian."""
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)
        rust_model = self._build_rust_model(expr, n_states)

        rng = np.random.default_rng(0)
        y = rng.standard_normal(n_states) * 0.1 + 0.5
        v = rng.standard_normal(n_states)
        t = 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        jvp_result = rust_model.rhs.jvp(t, y, inputs_arr, v)

        J = rust_model.jacobian(t, y, inputs_arr)
        assembled_result = np.asarray(J @ v)

        np.testing.assert_allclose(jvp_result, assembled_result, rtol=1e-9, atol=1e-12)

    def test_sparsity_pattern_structure(self):
        n_states = 20
        expr = self.build_diffusion_reaction_expr(n_states)
        rust_model = self._build_rust_model(expr, n_states)

        indptr, indices = rust_model.sparsity_pattern()
        indptr = np.array(indptr)
        indices = np.array(indices)

        assert len(indptr) == n_states + 1, "indptr should have n_states + 1 elements"
        assert indptr[0] == 0, "indptr should start at 0"
        assert indptr[-1] == len(indices), "indptr[-1] should equal nnz"

        assert np.all(indices >= 0), "all indices should be non-negative"
        assert np.all(indices < n_states), "all indices should be < n_states"

    def test_residual_evaluation(self):
        n_states = 20
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        f_fn, _, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        rng = np.random.default_rng(0)
        y = rng.standard_normal(n_states) * 0.1 + 0.5
        yp = rng.standard_normal(n_states) * 0.01
        t = 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        # Residual is M*y' - f with M = I.
        f_val = np.array(f_fn(t, y, *inputs.values())).flatten()
        expected_residual = yp - f_val

        rust_residual = np.array(rust_model.eval_residual(t, y, yp, inputs_arr))

        np.testing.assert_allclose(
            rust_residual, expected_residual, rtol=1e-10, atol=1e-14
        )

    @pytest.mark.parametrize("n_states", [10, 50, 100])
    def test_into_methods_match_allocating(self, n_states):
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)
        rust_model = self._build_rust_model(expr, n_states)

        rng = np.random.default_rng(0)
        y = rng.standard_normal(n_states) * 0.1 + 0.5
        v = rng.standard_normal(n_states)
        yp = rng.standard_normal(n_states) * 0.01
        t = 0.5
        cj = 1.0
        inputs_arr = np.array([inputs[name] for name in inputs])

        rhs_output = np.zeros(rust_model.output_len)
        residual_output = np.zeros(rust_model.n_states)

        rhs_alloc = np.array(rust_model.rhs(t, y, inputs_arr))
        rust_model.rhs.eval_into(t, y, inputs_arr, rhs_output)
        np.testing.assert_array_equal(rhs_output, rhs_alloc)

        # The Jacobian has no _into variant; two forms of (J - cj*M) @ v must agree.
        M = eye(n_states, format="csr")
        J = rust_model.jacobian(t, y, inputs_arr)
        jvp_via_matmul = np.array(J @ v - cj * (M @ v))
        jvp_via_sparse = np.array((J - cj * M) @ v)
        np.testing.assert_allclose(
            jvp_via_matmul, jvp_via_sparse, rtol=1e-14, atol=1e-14
        )

        res_alloc = np.array(rust_model.eval_residual(t, y, yp, inputs_arr))
        rust_model.eval_residual_into(t, y, yp, inputs_arr, residual_output)
        np.testing.assert_array_equal(residual_output, res_alloc)

    def test_varying_time_values(self):
        n_states = 20
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        f_fn, _, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        rng = np.random.default_rng(0)
        y = rng.standard_normal(n_states) * 0.1 + 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        for t in [0.0, 0.1, 0.5, 1.0, 10.0, 100.0]:
            casadi_result = np.array(f_fn(t, y, *inputs.values())).flatten()
            rust_result = np.array(rust_model.rhs(t, y, inputs_arr))
            np.testing.assert_allclose(
                rust_result,
                casadi_result,
                rtol=1e-10,
                atol=1e-14,
                err_msg=f"Mismatch at t={t}",
            )

    def test_model_properties(self):
        n_states = 25
        expr = self.build_diffusion_reaction_expr(n_states)
        rust_model = self._build_rust_model(expr, n_states)

        assert rust_model.n_states == n_states
        assert rust_model.output_len == n_states
        assert rust_model.n_colors > 0
        assert rust_model.n_colors <= n_states
