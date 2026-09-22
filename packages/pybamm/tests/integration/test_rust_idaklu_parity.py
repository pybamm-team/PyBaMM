"""E2E parity tests: Rust vs CasADi IDAKLU backends."""

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


def build_diffusion_reaction_expr(n_states: int):
    """Build ``f(y) = D * L @ y + k * exp(-y/T) * (1 - y)``."""
    sv = pybamm.StateVector(slice(0, n_states))
    D = pybamm.Scalar(1e-5)
    k = pybamm.Scalar(1.0)
    T = pybamm.Scalar(298.15)

    diag_data = [
        np.ones(n_states) * -2,
        np.ones(n_states - 1),
        np.ones(n_states - 1),
    ]
    laplacian = diags(diag_data, [0, -1, 1], format="csr")
    L = pybamm.Matrix(laplacian)

    diffusion = D * (L @ sv)
    reaction = k * pybamm.exp(-sv / T) * (pybamm.Scalar(1.0) - sv)

    return diffusion + reaction


def evaluate_expr_casadi(expr, n_states, t, y):
    t_sym = casadi.MX.sym("t")
    y_sym = casadi.MX.sym("y", n_states)
    y_dot_sym = casadi.MX.sym("y_dot", 0)

    casadi_symbols = {"t": t_sym, "y": y_sym, "y_dot": y_dot_sym, "inputs": {}}
    f_casadi = expr.to_casadi(t_sym, y_sym, y_dot_sym, {}, casadi_symbols)
    f_fn = casadi.Function("f", [t_sym, y_sym], [f_casadi])

    return np.array(f_fn(t, y)).flatten()


def evaluate_expr_rust(expr, n_states, t, y):
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

    return np.array(model.rhs(t, y, np.array([])))


class TestRustCasADiRHSParity:
    @pytest.mark.parametrize("n_states", [10, 50, 100])
    def test_rhs_parity_diffusion_reaction(self, n_states):
        expr = build_diffusion_reaction_expr(n_states)
        y = np.random.randn(n_states) * 0.1 + 0.5
        t = 0.0

        casadi_result = evaluate_expr_casadi(expr, n_states, t, y)
        rust_result = evaluate_expr_rust(expr, n_states, t, y)

        np.testing.assert_allclose(rust_result, casadi_result, rtol=1e-10, atol=1e-14)


class TestRustCasADiSolveParity:
    @pytest.fixture
    def spm_model_no_events(self):
        model = pybamm.lithium_ion.SPM()
        model.events = []
        return model

    @pytest.fixture
    def parameter_values(self):
        return pybamm.ParameterValues("Chen2020")

    def test_short_solve_parity(self, spm_model_no_events, parameter_values):
        t_eval = np.linspace(0, 100, 10)

        model_casadi = spm_model_no_events
        model_casadi.convert_to_format = "casadi"
        sol_casadi = pybamm.Simulation(
            model_casadi,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(),
        ).solve(t_eval)

        model_rust = spm_model_no_events.new_copy()
        model_rust.convert_to_format = "rust"
        sol_rust = pybamm.Simulation(
            model_rust,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(),
        ).solve(t_eval)

        np.testing.assert_allclose(sol_rust.y, sol_casadi.y, rtol=1e-5, atol=1e-8)

    def test_full_discharge_parity(self, spm_model_no_events, parameter_values):
        t_eval = np.linspace(0, 3600, 100)

        model_casadi = spm_model_no_events
        model_casadi.convert_to_format = "casadi"
        sol_casadi = pybamm.Simulation(
            model_casadi,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(),
        ).solve(t_eval)

        model_rust = spm_model_no_events.new_copy()
        model_rust.convert_to_format = "rust"
        sol_rust = pybamm.Simulation(
            model_rust,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(),
        ).solve(t_eval)

        np.testing.assert_allclose(sol_rust.y, sol_casadi.y, rtol=1e-5, atol=1e-8)
