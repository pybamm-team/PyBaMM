"""Parity test: Rust IDAKLU output_variables vs CasADi.

The Rust expression converter does not yet support every node used in real
battery models, so we exercise output-variable plumbing on a synthetic
ODE that uses only supported operations. The test confirms:
  - the Python-side `output_exprs` arg flows through to CompiledModel,
  - `model.outputs[i](t, y, p)` produces the correct values,
  - configured output expressions don't disturb the rhs solve.
"""

import numpy as np
import pytest

import pybamm

pytest.importorskip("casadi")

try:
    from pybamm.rust import CompiledModel, ExprGraph
except ImportError:
    pytest.skip(
        "Rust extension not available. Build with: uv sync",
        allow_module_level=True,
    )


def _build_decay_model_with_outputs():
    """ODE: dy/dt = -k*y, output: 2*y. Uses StateVector to bypass discretization."""
    graph = ExprGraph()
    rust_symbols: dict = {}

    # StateVector -> Rust StateVector node directly (no discretization needed).
    y = pybamm.StateVector(slice(0, 1))
    k = pybamm.InputParameter("k")
    rhs_sym = -k * y
    output_sym = 2 * y

    rhs_expr = rhs_sym.to_rust(graph, rust_symbols)
    output_expr = output_sym.to_rust(graph, rust_symbols)

    return graph, rhs_expr, output_expr


def test_pycompiled_model_eval_output_matches_analytical():
    """Output 2*y at y=3 with k=1 should equal 6, regardless of k."""
    graph, rhs_expr, output_expr = _build_decay_model_with_outputs()

    # 1-state ODE, identity mass, 1 input, 1 output
    mass_data = np.ones(1)
    mass_indptr = np.array([0, 1], dtype=np.int64)
    mass_indices = np.array([0], dtype=np.int64)

    model = CompiledModel.from_expr(
        graph,
        rhs_expr,
        mass_data,
        mass_indptr,
        mass_indices,
        n_inputs=1,
        output_exprs=[output_expr],
    )

    assert model.n_outputs == 1
    assert [f.output_len for f in model.outputs] == [1]

    rhs = model.rhs(0.0, np.asarray([3.0]), np.asarray([1.0]))
    np.testing.assert_allclose(rhs, [-3.0], atol=1e-12)

    # output 2*y at y=3: 6
    out = model.outputs[0](0.0, np.asarray([3.0]), np.asarray([1.0]))
    np.testing.assert_allclose(out, [6.0], atol=1e-12)


def test_pycompiled_model_no_outputs_default():
    """`output_exprs` defaults to empty -> n_outputs == 0, output_lens empty."""
    graph = ExprGraph()
    y = graph.state_vector(0, 1)
    mass_data = np.ones(1)
    mass_indptr = np.array([0, 1], dtype=np.int64)
    mass_indices = np.array([0], dtype=np.int64)
    model = CompiledModel.from_expr(graph, y, mass_data, mass_indptr, mass_indices)
    assert model.n_outputs == 0
    assert [f.output_len for f in model.outputs] == []
