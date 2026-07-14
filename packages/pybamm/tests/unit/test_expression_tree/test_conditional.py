import casadi
import numpy as np
import pytest
import scipy.sparse

import pybamm
from pybamm.expression_tree.operations.serialise import (
    convert_symbol_from_json,
    convert_symbol_to_json,
)


class TestConditional:
    def test_evaluate_scalar_branches(self):
        selector = pybamm.InputParameter("selector")
        expr = pybamm.Conditional(selector, pybamm.Scalar(3), pybamm.Scalar(7))

        assert expr.evaluate(inputs={"selector": 1}) == 3
        assert expr.evaluate(inputs={"selector": 2}) == 7
        assert expr.evaluate(inputs={"selector": 0}) == 0
        assert expr.evaluate(inputs={"selector": 0.5}) == 0
        assert expr.evaluate(inputs={"selector": 1.5}) == 0

    def test_evaluate_vector_branches_and_shape(self):
        selector = pybamm.InputParameter("selector")
        expr = pybamm.Conditional(
            selector,
            pybamm.Vector(np.array([1, 2])),
            pybamm.Vector(np.array([3, 4])),
        )

        np.testing.assert_array_equal(
            expr.evaluate(inputs={"selector": 2}),
            np.array([[3], [4]]),
        )
        np.testing.assert_array_equal(
            expr.evaluate(inputs={"selector": -1}),
            np.zeros((2, 1)),
        )
        assert expr.shape == (2, 1)

    def test_validation(self):
        with pytest.raises(ValueError, match="at least one branch"):
            pybamm.Conditional(pybamm.Scalar(1))

        with pytest.raises(ValueError, match="selector must evaluate to a scalar"):
            pybamm.Conditional(
                pybamm.Vector(np.array([1, 2])),
                pybamm.Scalar(1),
            )

        with pytest.raises(ValueError, match="must all have the same shape"):
            pybamm.Conditional(
                pybamm.Scalar(1),
                pybamm.Scalar(1),
                pybamm.Vector(np.array([1, 2])),
            )

        with pytest.raises(
            pybamm.DomainError, match="children must have same or empty"
        ):
            pybamm.Conditional(
                pybamm.Scalar(1),
                pybamm.Variable("a", domain=["negative electrode"]),
                pybamm.Variable("b", domain=["positive electrode"]),
            )

    def test_diff_and_jacobian(self):
        x = pybamm.StateVector(slice(0, 1))
        selector = pybamm.InputParameter("selector")
        expr = pybamm.Conditional(selector, x**2, x**3)
        deriv = expr.diff(x)

        np.testing.assert_allclose(
            deriv.evaluate(inputs={"selector": 1}, y=np.array([2.0])),
            np.array([[4.0]]),
        )
        np.testing.assert_allclose(
            deriv.evaluate(inputs={"selector": 2}, y=np.array([2.0])),
            np.array([[12.0]]),
        )

        jac_expr = pybamm.Conditional(selector, x**2, x**3).jac(x)
        np.testing.assert_allclose(
            jac_expr.evaluate(inputs={"selector": 1}, y=np.array([2.0])),
            np.array([[4.0]]),
        )
        np.testing.assert_allclose(
            jac_expr.evaluate(inputs={"selector": 2}, y=np.array([2.0])),
            np.array([[12.0]]),
        )

    def test_copy_equation_and_serialisation(self):
        selector = pybamm.InputParameter("selector")
        expr = pybamm.Conditional(selector, pybamm.Scalar(1), pybamm.Scalar(2))

        copied = expr.create_copy()
        assert copied == expr
        assert str(expr.to_equation()) == "Conditional(selector, 1.0, 2.0)"

        json_data = convert_symbol_to_json(expr)
        rebuilt = convert_symbol_from_json(json_data)
        assert isinstance(rebuilt, pybamm.Conditional)
        assert rebuilt.evaluate(inputs={"selector": 2}) == 2

    def test_from_json_and_str(self):
        selector = pybamm.InputParameter("selector")
        expr = pybamm.Conditional._from_json(
            {"children": [selector, pybamm.Scalar(1), pybamm.Scalar(2)]}
        )

        assert isinstance(expr, pybamm.Conditional)
        assert str(expr) == "conditional(selector, 1.0, 2.0)"

    def test_create_copy_validation_and_selector_must_be_scalar(self):
        expr = pybamm.Conditional(
            pybamm.InputParameter("selector"),
            pybamm.Scalar(3),
            pybamm.Scalar(4),
        )

        with pytest.raises(
            ValueError, match="Conditional must have a selector and at least one branch"
        ):
            expr.create_copy([pybamm.Scalar(1)])

        with pytest.raises(
            ValueError, match="Conditional selector must evaluate to a scalar"
        ):
            expr._coerce_selector_value(np.array([1, 2]))

    def test_zero_like_numeric_and_sparse_branch_shape(self):
        assert pybamm.Conditional._zero_like(5) == 0

        selector = pybamm.InputParameter("selector")
        sparse_branch = pybamm.Matrix(scipy.sparse.csr_matrix([[1, 0], [0, 1]]))
        expr = pybamm.Conditional(selector, sparse_branch)

        shape = expr.evaluate_for_shape()
        assert scipy.sparse.issparse(shape)
        assert shape.shape == (2, 2)

        out = expr.evaluate(inputs={"selector": 0})
        assert scipy.sparse.issparse(out)
        assert out.shape == (2, 2)
        assert out.nnz == 0

    def test_evaluates_on_edges(self):
        expr_on_edges = pybamm.Conditional(
            pybamm.Scalar(1),
            pybamm.PrimaryBroadcastToEdges(pybamm.Scalar(1), ["negative electrode"]),
        )
        assert expr_on_edges.evaluates_on_edges("primary")

        expr_not_on_edges = pybamm.Conditional(
            pybamm.Scalar(1),
            pybamm.PrimaryBroadcast(pybamm.Scalar(1), ["negative electrode"]),
        )
        assert not expr_not_on_edges.evaluates_on_edges("primary")

    def test_evaluator_python(self):
        selector = pybamm.InputParameter("selector")
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        expr = pybamm.Conditional(selector, a + b, a - b)

        evaluator = pybamm.EvaluatorPython(expr)
        y = np.array([5.0, 2.0])
        np.testing.assert_allclose(
            evaluator(y=y, inputs={"selector": 1}), np.array([[7.0]])
        )
        np.testing.assert_allclose(
            evaluator(y=y, inputs={"selector": 2}), np.array([[3.0]])
        )
        np.testing.assert_allclose(evaluator(y=y, inputs={"selector": 0}), 0)

    def test_to_casadi(self):
        selector = pybamm.InputParameter("selector")
        y = pybamm.StateVector(slice(0, 1))
        expr = pybamm.Conditional(selector, 2 * y, 3 * y)

        casadi_y = casadi.MX.sym("y", 1)
        casadi_selector = casadi.MX.sym("selector")
        casadi_expr = expr.to_casadi(y=casadi_y, inputs={"selector": casadi_selector})
        f = casadi.Function("f", [casadi_y, casadi_selector], [casadi_expr])

        np.testing.assert_allclose(np.array(f([4.0], 1.0)).reshape(-1), np.array([8.0]))
        np.testing.assert_allclose(
            np.array(f([4.0], 2.0)).reshape(-1), np.array([12.0])
        )
        np.testing.assert_allclose(np.array(f([4.0], 0.5)).reshape(-1), np.array([0.0]))
