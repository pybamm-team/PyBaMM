# tests/conftest_rust.py
"""Pytest fixtures for Rust expression backend testing."""

import numpy as np
import pytest

import pybamm


@pytest.fixture
def rust_backend(monkeypatch):
    """Make newly constructed models default to the Rust backend."""
    monkeypatch.setattr(pybamm.BaseModel, "_DEFAULT_CONVERT_TO_FORMAT", "rust")
    yield


def evaluate_with_rust(expr, t=0.0, y=None, y_dot=None, inputs=None):
    """Evaluate a PyBaMM expression using the Rust backend.

    Parameters
    ----------
    expr : pybamm.Symbol
        The expression to evaluate
    t : float
        Time value
    y : array-like, optional
        State vector
    y_dot : array-like, optional
        State vector derivative
    inputs : dict, optional
        Input parameters as name->value dict

    Returns
    -------
    numpy.ndarray or float
        The evaluation result
    """
    from pybamm.rust import ExprGraph

    graph = ExprGraph()
    rust_symbols = {}
    rust_expr = expr.to_rust(graph, rust_symbols)

    # Convert inputs dict to ordered list - order matches registration during to_rust()
    # We need to extract input parameter names from the expression tree
    if inputs is None:
        inputs_list = []
    else:
        # Get input parameter names in the order they appear in the expression
        input_params = [
            node.name
            for node in expr.pre_order()
            if isinstance(node, pybamm.InputParameter)
        ]
        # Deduplicate, preserving order.
        seen = set()
        unique_params = []
        for name in input_params:
            if name not in seen:
                seen.add(name)
                unique_params.append(name)
        inputs_list = [inputs.get(name, 0.0) for name in unique_params]

    y_np = (
        np.asarray(y, dtype=np.float64).ravel()
        if y is not None
        else np.array([], dtype=np.float64)
    )
    y_dot_np = (
        np.asarray(y_dot, dtype=np.float64).ravel()
        if y_dot is not None
        else np.array([], dtype=np.float64)
    )

    # Determine scalar vs array by checking PyBaMM's evaluation result shape
    pybamm_result = expr.evaluate(t=t, y=y, y_dot=y_dot, inputs=inputs)
    is_scalar = np.isscalar(pybamm_result) or (
        hasattr(pybamm_result, "shape") and pybamm_result.size == 1
    )

    if is_scalar:
        return graph.eval_to_float(
            rust_expr, t, y_np.tolist(), y_dot_np.tolist(), inputs_list
        )
    else:
        return np.array(graph.eval_to_array(rust_expr, t, y_np, y_dot_np, inputs_list))


def evaluate_with_casadi(expr, t=0.0, y=None, y_dot=None, inputs=None):
    """Evaluate a PyBaMM expression using the CasADi backend.

    Parameters
    ----------
    expr : pybamm.Symbol
        The expression to evaluate
    t : float
        Time value
    y : array-like, optional
        State vector
    y_dot : array-like, optional
        State vector derivative
    inputs : dict, optional
        Input parameters as name->value dict

    Returns
    -------
    numpy.ndarray or float
        The evaluation result
    """
    import casadi

    # Determine sizes
    y_size = 0 if y is None else np.asarray(y).size
    y_dot_size = 0 if y_dot is None else np.asarray(y_dot).size

    # Create CasADi symbols
    t_sym = casadi.MX.sym("t")
    y_sym = casadi.MX.sym("y", y_size) if y_size > 0 else casadi.MX.sym("y", 0)
    y_dot_sym = (
        casadi.MX.sym("y_dot", y_dot_size)
        if y_dot_size > 0
        else casadi.MX.sym("y_dot", 0)
    )

    # Build inputs symbols
    if inputs is None:
        inputs = {}
    inputs_sym = {name: casadi.MX.sym(name) for name in inputs}

    casadi_symbols = {"t": t_sym, "y": y_sym, "y_dot": y_dot_sym, "inputs": inputs_sym}

    # Convert expression
    casadi_expr = expr.to_casadi(t_sym, y_sym, y_dot_sym, inputs_sym, casadi_symbols)

    # Build function inputs list
    func_inputs = [t_sym, y_sym, y_dot_sym, *inputs_sym.values()]
    func = casadi.Function("f", func_inputs, [casadi_expr])

    y_val = np.array([]) if y is None else np.asarray(y).flatten()
    y_dot_val = np.array([]) if y_dot is None else np.asarray(y_dot).flatten()
    input_vals = [inputs[name] for name in inputs_sym]

    result = func(t, y_val, y_dot_val, *input_vals)
    return np.asarray(result).flatten()


@pytest.fixture
def dual_backend_compare():
    """Fixture that compares Rust and CasADi evaluation results.

    Usage:
        def test_something(dual_backend_compare):
            expr = pybamm.Scalar(1.0) + pybamm.Scalar(2.0)
            dual_backend_compare(expr)  # asserts results match
    """

    def _compare(expr, t=0.0, y=None, y_dot=None, inputs=None, rtol=1e-10, atol=1e-14):
        rust_result = evaluate_with_rust(expr, t, y, y_dot, inputs)
        casadi_result = evaluate_with_casadi(expr, t, y, y_dot, inputs)

        np.testing.assert_allclose(
            rust_result,
            casadi_result,
            rtol=rtol,
            atol=atol,
            err_msg=f"Rust vs CasADi mismatch for {type(expr).__name__}",
        )
        return rust_result

    return _compare
