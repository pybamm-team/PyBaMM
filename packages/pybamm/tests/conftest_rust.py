"""Pytest fixtures for Rust expression backend testing."""

import numpy as np
import pytest
from scipy.sparse import issparse

import pybamm


def _reads_y_dot(expr):
    return any(isinstance(node, pybamm.StateVectorDot) for node in expr.pre_order())


def _casadi_functions(expr, n_states, input_sizes):
    """CasADi value, df/dy, df/dp and jvp functions of ``expr``, inputs stacked in
    the order of ``input_sizes``."""
    import casadi

    t = casadi.MX.sym("t")
    y = casadi.MX.sym("y", n_states)
    y_dot = casadi.MX.sym("y_dot", n_states)
    inputs = {name: casadi.MX.sym(name, size) for name, size in input_sizes.items()}
    p = casadi.vertcat(*inputs.values()) if inputs else casadi.MX.sym("p", 0)
    v = casadi.MX.sym("v", n_states)
    value = expr.to_casadi(t, y, y_dot, inputs)
    functions = {"value": casadi.Function("f", [t, y, y_dot, p], [value])}
    if not _reads_y_dot(expr):
        functions["jac_y"] = casadi.Function(
            "jac_y", [t, y, p], [casadi.jacobian(value, y)]
        )
        functions["jac_p"] = casadi.Function(
            "jac_p", [t, y, p], [casadi.jacobian(value, p)]
        )
        functions["jvp"] = casadi.Function(
            "jvp", [t, y, p, v], [casadi.jtimes(value, y, v)]
        )
    return functions


@pytest.fixture
def dual_backend_compare():
    """Return a callable asserting the Rust backend matches ``evaluate()`` and CasADi.

    The value is checked against both. Unless ``expr`` reads ``y_dot``, the
    jacobian with respect to ``y`` and a jvp are checked against CasADi, and so is
    the jacobian with respect to the inputs when every input is scalar. Pass
    ``check_evaluate=False`` where ``evaluate()`` follows a different convention
    from the compiled backends.
    """

    def _compare(
        expr,
        t=0.0,
        y=None,
        y_dot=None,
        inputs=None,
        rtol=1e-10,
        atol=1e-14,
        check_evaluate=True,
    ):
        from pybamm.rust import ExprGraph

        inputs = inputs or {}
        y = np.zeros(0) if y is None else np.asarray(y, dtype=np.float64).ravel()
        if y_dot is not None:
            y_dot = np.asarray(y_dot, dtype=np.float64).ravel()
            y = y if y.size else np.zeros(y_dot.size)
        n_states = y.size
        y_dot_value = np.zeros(n_states) if y_dot is None else y_dot

        graph = ExprGraph()
        compiled = graph.compile(expr.to_rust(graph, {}), n_states=n_states)
        names = compiled.input_names
        # Inputs register in order of first appearance in the tree
        assert names == tuple(
            dict.fromkeys(
                node.name
                for node in expr.pre_order()
                if isinstance(node, pybamm.InputParameter)
            )
        )
        p = {name: inputs[name] for name in names}
        p_stacked = np.concatenate(
            [np.ravel(inputs[name]).astype(np.float64) for name in names] + [[]]
        )
        casadi_fns = _casadi_functions(
            expr, n_states, {name: np.size(inputs[name]) for name in names}
        )

        rust_value = np.asarray(compiled(t, y, p, y_dot=y_dot_value)).ravel()
        casadi_value = np.asarray(
            casadi_fns["value"](t, y, y_dot_value, p_stacked)
        ).ravel()
        np.testing.assert_allclose(
            rust_value, casadi_value, rtol=rtol, atol=atol, err_msg="value vs CasADi"
        )
        if check_evaluate:
            expected = expr.evaluate(
                t=t,
                y=y.reshape(-1, 1) if n_states else None,
                y_dot=None if y_dot is None else y_dot.reshape(-1, 1),
                inputs=inputs,
            )
            expected = expected.toarray() if issparse(expected) else expected
            np.testing.assert_allclose(
                rust_value,
                np.ravel(expected),
                rtol=rtol,
                atol=atol,
                err_msg="value vs evaluate()",
            )

        if "jac_y" in casadi_fns:
            if n_states:
                np.testing.assert_allclose(
                    compiled.jacobian("y")(t, y, p).toarray(),
                    np.asarray(casadi_fns["jac_y"](t, y, p_stacked)),
                    rtol=rtol,
                    atol=atol,
                    err_msg="df/dy vs CasADi",
                )
                direction = np.linspace(0.5, 1.5, n_states)
                np.testing.assert_allclose(
                    compiled.jvp(t, y, p, direction),
                    np.asarray(casadi_fns["jvp"](t, y, p_stacked, direction)).ravel(),
                    rtol=rtol,
                    atol=atol,
                    err_msg="jvp vs CasADi",
                )
            # Rust's df/dp has a column per input, CasADi's one per input entry
            if names and all(np.size(inputs[name]) == 1 for name in names):
                np.testing.assert_allclose(
                    compiled.jacobian("p")(t, y, p).toarray(),
                    np.asarray(casadi_fns["jac_p"](t, y, p_stacked)),
                    rtol=rtol,
                    atol=atol,
                    err_msg="df/dp vs CasADi",
                )
        return rust_value

    return _compare
