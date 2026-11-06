#
# Tests for compiling expressions into functions of named inputs
#
import numpy as np
import pytest

import pybamm
from pybamm.expression_tree.operations.input_jacobian import compile_input_jacobian

A, B = 0.3, 1.2


def _symbols():
    a = pybamm.InputParameter("a")
    b = pybamm.InputParameter("b")
    x = np.linspace(0, 1, 11)
    y = np.linspace(0, 2, 21)
    # Linear interpolants of linear data are exact, with exact derivatives
    table = 2 * x[:, np.newaxis] + 3 * y[np.newaxis, :]
    return [
        pybamm.exp(a) * b,
        pybamm.tanh(a) + pybamm.Constant(2.0, "two"),
        pybamm.minimum(a, b),
        pybamm.Interpolant(x, 4 * x, a),
        pybamm.Interpolant((x, y), table, (a, b)),
    ]


def _expected():
    values = [np.exp(A) * B, np.tanh(A) + 2, A, 4 * A, 2 * A + 3 * B]
    jacobian = [
        [np.exp(A) * B, np.exp(A)],
        [1 - np.tanh(A) ** 2, 0],
        [1, 0],
        [4, 0],
        [2, 3],
    ]
    return np.array(values), np.array(jacobian)


class TestInputJacobian:
    @pytest.mark.parametrize("convert_to_format", [None, "casadi", "python", "jax"])
    def test_values_and_jacobian(self, convert_to_format):
        function = compile_input_jacobian(
            _symbols(), ["a", "b"], convert_to_format=convert_to_format
        )
        values, jacobian = function(np.array([A, B]))
        expected_values, expected_jacobian = _expected()
        np.testing.assert_allclose(values, expected_values, rtol=1e-12)
        np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1e-12, atol=1e-14)

    def test_columns_follow_input_names(self):
        # "c" enters no symbol, so its column is zero
        function = compile_input_jacobian(_symbols(), ["c", "b", "a"])
        values, jacobian = function(np.array([5.0, B, A]))
        expected_values, expected_jacobian = _expected()
        np.testing.assert_allclose(values, expected_values, rtol=1e-12)
        np.testing.assert_allclose(
            jacobian,
            np.hstack([np.zeros((5, 1)), expected_jacobian[:, ::-1]]),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_cached(self):
        symbols = _symbols()
        function = compile_input_jacobian(symbols, ["a", "b"])
        assert compile_input_jacobian(_symbols(), ("a", "b")) is function
        assert compile_input_jacobian(symbols, ["b", "a"]) is not function
