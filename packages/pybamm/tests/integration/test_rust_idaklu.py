# tests/integration/test_rust_idaklu.py
"""Integration tests verifying Rust evaluator produces IDAKLU-compatible Jacobians.

These tests validate that the Rust `CompiledModel` produces correct Jacobian
values that match CasADi. This is the foundation for integrating Rust evaluation
into the IDAKLU solver pipeline.

The tests focus on:
1. RHS evaluation (f(t, y)) matching between Rust and CasADi
2. Jacobian-vector product (df/dy @ v - cj*M @ v) matching
3. Assembled Jacobian correctness via consistency with JVP
"""

import numpy as np
import pytest
from scipy.sparse import diags, eye

import pybamm

casadi = pytest.importorskip("casadi")

# Try to import Rust extension
try:
    from pybamm.rust import CompiledModel, ExprGraph
except ImportError:
    pytest.skip(
        "Rust extension not available. Build with: uv sync",
        allow_module_level=True,
    )


class TestRustIDAKLUJacobianParity:
    """Tests verifying Rust Jacobian evaluation matches CasADi for IDAKLU."""

    @staticmethod
    def build_diffusion_reaction_expr(n_states: int):
        """Build a realistic diffusion-reaction expression like in battery models.

        f(y) = D * L @ y + k * exp(-y/T) * (1 - y)

        where:
        - L is a tridiagonal Laplacian matrix (diffusion)
        - k * exp(-y/T) * (1 - y) is a Butler-Volmer-like reaction term

        This mimics the structure of SPM/DFN equations.
        """
        sv = pybamm.StateVector(slice(0, n_states))
        t = pybamm.Time()
        D = pybamm.InputParameter("diffusivity")  # Diffusion coefficient
        k = pybamm.InputParameter("rate_constant")  # Reaction rate
        T = pybamm.InputParameter("temperature")

        # Tridiagonal Laplacian (FD discretization of d^2/dx^2)
        diag_data = [
            np.ones(n_states) * -2,
            np.ones(n_states - 1),
            np.ones(n_states - 1),
        ]
        laplacian = diags(diag_data, [0, -1, 1], format="csr")
        L = pybamm.Matrix(laplacian)

        # Diffusion term: D * L @ y
        diffusion = D * (L @ sv)

        # Reaction term: k * exp(-y/T) * (1 - y) * (1 + 0.01*t)
        reaction = (
            k
            * pybamm.exp(-sv / T)
            * (pybamm.Scalar(1.0) - sv)
            * (pybamm.Scalar(1.0) + pybamm.Scalar(0.01) * t)
        )

        return diffusion + reaction

    @staticmethod
    def build_stiff_ode_expr(n_states: int):
        """Build a stiff ODE system like in electrochemistry.

        f(y) = -A @ y + nonlinear_source(y)

        where A has eigenvalues spanning several orders of magnitude.
        """
        sv = pybamm.StateVector(slice(0, n_states))
        t = pybamm.Time()
        alpha = pybamm.InputParameter("alpha")
        beta = pybamm.InputParameter("beta")

        # Stiff matrix: band structure with varying eigenvalues
        diag_main = np.linspace(1.0, 1000.0, n_states)  # Eigenvalues 1 to 1000
        diag_off = np.ones(n_states - 1) * 0.1
        A_sparse = diags([diag_main, -diag_off, -diag_off], [0, -1, 1], format="csr")
        A = pybamm.Matrix(A_sparse)

        # Linear decay
        linear = -alpha * (A @ sv)

        # Nonlinear source: beta * tanh(y) * exp(-0.01 * y^2)
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

        # RHS function
        func_inputs = [t_sym, y_sym, *inputs_sym.values()]
        f_fn = casadi.Function("f", func_inputs, [f_casadi])

        # Full Jacobian function
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

        # Mass matrix (identity for ODE)
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
        """Test that Rust RHS evaluation matches CasADi for diffusion-reaction."""
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        # Build models
        f_fn, _, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        # Test points
        y = np.random.randn(n_states) * 0.1 + 0.5
        t = 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        casadi_result = np.array(f_fn(t, y, *inputs.values())).flatten()
        rust_result = np.array(rust_model.rhs(t, y, inputs_arr))

        np.testing.assert_allclose(rust_result, casadi_result, rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("n_states", [10, 50, 100])
    def test_rhs_parity_stiff_ode(self, n_states):
        """Test that Rust RHS evaluation matches CasADi for stiff ODE."""
        inputs = {"alpha": 1.0, "beta": 0.5}
        expr = self.build_stiff_ode_expr(n_states)

        # Build models
        f_fn, _, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        # Test points
        y = np.random.randn(n_states) * 0.1 + 0.5
        t = 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        casadi_result = np.array(f_fn(t, y, *inputs.values())).flatten()
        rust_result = np.array(rust_model.rhs(t, y, inputs_arr))

        np.testing.assert_allclose(rust_result, casadi_result, rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("n_states", [10, 50, 100])
    def test_jac_mul_parity(self, n_states):
        """Test that Rust Jacobian-vector product matches CasADi."""
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        # Build models
        _, _, jac_action_fn = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        # Test points
        y = np.random.randn(n_states) * 0.1 + 0.5
        v = np.random.randn(n_states)
        t = 0.5
        cj = 1.0
        inputs_arr = np.array([inputs[name] for name in inputs])

        # Evaluate: J @ v - cj * M @ v; M = I for ODE tests
        casadi_result = np.array(jac_action_fn(t, y, *inputs.values(), cj, v)).flatten()
        J = rust_model.jacobian(t, y, inputs_arr)
        M = eye(n_states, format="csr")
        rust_result = np.array(J @ v - cj * (M @ v))

        np.testing.assert_allclose(rust_result, casadi_result, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("n_states", [10, 50])
    def test_assembled_jacobian_parity(self, n_states):
        """Test that Rust assembled Jacobian matches CasADi full Jacobian."""
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        # Build models
        _, jac_fn, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        # Test points
        y = np.random.randn(n_states) * 0.1 + 0.5
        t = 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        # CasADi full Jacobian
        casadi_jac = np.array(jac_fn(t, y, *inputs.values()))

        # Rust assembled Jacobian (CSC via bundle accessor); model.jacobian = pure df/dy
        J = rust_model.jacobian(t, y, inputs_arr)
        rust_jac = J.toarray()

        np.testing.assert_allclose(rust_jac, casadi_jac, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("n_states", [10, 50])
    def test_assembled_jacobian_with_cj(self, n_states):
        """Test that assembled Jacobian with cj correctly computes df/dy - cj*M."""
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        # Build models
        _, jac_fn, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        # Test points
        y = np.random.randn(n_states) * 0.1 + 0.5
        t = 0.5
        cj = 1.5  # Non-trivial cj value
        inputs_arr = np.array([inputs[name] for name in inputs])

        # Expected: df/dy - cj*M = df/dy - cj*I
        casadi_jac = np.array(jac_fn(t, y, *inputs.values()))
        expected_jac = casadi_jac - cj * np.eye(n_states)

        # Rust Jacobian (CSC) minus cj*M; M = I for ODE tests
        J = rust_model.jacobian(t, y, inputs_arr)
        M = eye(n_states, format="csr")
        rust_jac = (J - cj * M).toarray()

        np.testing.assert_allclose(rust_jac, expected_jac, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("n_states", [10, 50])
    def test_jacobian_jvp_consistency(self, n_states):
        """Cross-check two independent Rust kernels on J @ v.

        The matrix-free tangent JVP (rhs view, one tangent sweep with the
        caller's seed) against the colored-assembly Jacobian (jacobian
        view, n_colors sweeps + scatter). A coloring/scatter bug that
        affects only one path fails this; J itself is pinned against
        casadi in test_assembled_jacobian_parity.
        """
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)
        rust_model = self._build_rust_model(expr, n_states)

        # Test points
        y = np.random.randn(n_states) * 0.1 + 0.5
        v = np.random.randn(n_states)
        t = 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        # Kernel 1: matrix-free tangent JVP via the rhs view
        jvp_result = rust_model.rhs.jvp(t, y, inputs_arr, v)

        # Kernel 2: colored assembly via the jacobian view, then matmul
        J = rust_model.jacobian(t, y, inputs_arr)
        assembled_result = np.asarray(J @ v)

        np.testing.assert_allclose(jvp_result, assembled_result, rtol=1e-9, atol=1e-12)

    def test_sparsity_pattern_structure(self):
        """Test that Rust model correctly reports sparsity pattern."""
        n_states = 20
        expr = self.build_diffusion_reaction_expr(n_states)
        rust_model = self._build_rust_model(expr, n_states)

        # Get sparsity pattern
        indptr, indices = rust_model.sparsity_pattern()
        indptr = np.array(indptr)
        indices = np.array(indices)

        # Basic structural checks
        assert len(indptr) == n_states + 1, "indptr should have n_states + 1 elements"
        assert indptr[0] == 0, "indptr should start at 0"
        assert indptr[-1] == len(indices), "indptr[-1] should equal nnz"

        # All indices should be valid column indices
        assert np.all(indices >= 0), "all indices should be non-negative"
        assert np.all(indices < n_states), "all indices should be < n_states"

    def test_residual_evaluation(self):
        """Test DAE residual evaluation r = M*y' - f(t,y)."""
        n_states = 20
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        # Build models
        f_fn, _, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        # Test points
        y = np.random.randn(n_states) * 0.1 + 0.5
        yp = np.random.randn(n_states) * 0.01  # y' (time derivative)
        t = 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        # Expected residual: M*y' - f = I*y' - f = y' - f (for M = I)
        f_val = np.array(f_fn(t, y, *inputs.values())).flatten()
        expected_residual = yp - f_val

        rust_residual = np.array(rust_model.eval_residual(t, y, yp, inputs_arr))

        np.testing.assert_allclose(
            rust_residual, expected_residual, rtol=1e-10, atol=1e-14
        )

    @pytest.mark.parametrize("n_states", [10, 50, 100])
    def test_into_methods_match_allocating(self, n_states):
        """Test _into methods match allocating versions (rhs, residual).

        The jacobian block is arithmetic-only: the new bundle jacobian has
        no _into variant (assemble_jacobian_csc_into covers the solver path).
        """
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)
        rust_model = self._build_rust_model(expr, n_states)

        y = np.random.randn(n_states) * 0.1 + 0.5
        v = np.random.randn(n_states)
        yp = np.random.randn(n_states) * 0.01
        t = 0.5
        cj = 1.0
        inputs_arr = np.array([inputs[name] for name in inputs])

        # Pre-allocate output buffers
        rhs_output = np.zeros(rust_model.output_len)
        residual_output = np.zeros(rust_model.n_states)

        # Compare rhs allocating vs eval_into
        rhs_alloc = np.array(rust_model.rhs(t, y, inputs_arr))
        rust_model.rhs.eval_into(t, y, inputs_arr, rhs_output)
        np.testing.assert_array_equal(rhs_output, rhs_alloc)

        # Jacobian consistency: two equivalent forms of (J - cj*M) @ v must agree
        # to within floating-point rounding (different summation order → ULP diffs).
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
        """Test that time-dependent expressions evaluate correctly at multiple times."""
        n_states = 20
        inputs = {"diffusivity": 1e-5, "rate_constant": 1.0, "temperature": 298.15}
        expr = self.build_diffusion_reaction_expr(n_states)

        # Build models
        f_fn, _, _ = self._build_casadi_functions(expr, n_states, inputs)
        rust_model = self._build_rust_model(expr, n_states)

        y = np.random.randn(n_states) * 0.1 + 0.5
        inputs_arr = np.array([inputs[name] for name in inputs])

        # Test at multiple time values
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
        """Test that model properties are correctly reported."""
        n_states = 25
        expr = self.build_diffusion_reaction_expr(n_states)
        rust_model = self._build_rust_model(expr, n_states)

        assert rust_model.n_states == n_states
        assert rust_model.output_len == n_states
        # n_colors should be positive and <= n_states
        assert rust_model.n_colors > 0
        assert rust_model.n_colors <= n_states
