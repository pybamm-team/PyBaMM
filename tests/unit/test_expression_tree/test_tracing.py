#
# Tests for the tracing module
#
import threading

import numpy as np
import pytest

import pybamm
from pybamm.expression_tree.tracing import is_tracing, tracing


class TestTracingContextManager:
    def test_default_is_not_tracing(self):
        assert is_tracing() is False

    def test_tracing_enables_flag(self):
        assert is_tracing() is False
        with tracing():
            assert is_tracing() is True
        assert is_tracing() is False

    def test_tracing_restores_on_exception(self):
        with pytest.raises(RuntimeError, match="boom"):
            with tracing():
                assert is_tracing() is True
                raise RuntimeError("boom")
        assert is_tracing() is False

    def test_nested_tracing(self):
        assert is_tracing() is False
        with tracing():
            assert is_tracing() is True
            with tracing():
                assert is_tracing() is True
            assert is_tracing() is True
        assert is_tracing() is False

    def test_tracing_is_thread_local(self):
        """ContextVar state does not leak across threads."""
        seen_in_thread = []

        def worker():
            seen_in_thread.append(is_tracing())

        with tracing():
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        assert seen_in_thread == [False]

    def test_accessible_from_pybamm(self):
        assert pybamm.tracing is tracing
        assert pybamm.is_tracing is is_tracing


class TestTracingSuppressesSimplifications:
    """Each test constructs an expression that triggers a specific
    FP-reordering simplification outside tracing, and verifies that
    the simplification is suppressed inside tracing.
    """

    def test_power_distribution_suppressed(self):
        """(a * b) ** c -> (a**c) * (b**c) when a is constant."""
        a = pybamm.Scalar(2)
        b = pybamm.Parameter("b")
        c = pybamm.Scalar(3)

        result_normal = (a * b) ** c
        assert isinstance(result_normal, pybamm.Multiplication)

        with tracing():
            result_traced = (a * b) ** c
        assert isinstance(result_traced, pybamm.Power)

    def test_add_reassociation_suppressed(self):
        """a + (b + c) -> (a + b) + c when a, b are constant."""
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        c = pybamm.Parameter("c")

        result_normal = a + (b + c)
        assert isinstance(result_normal.left, pybamm.Scalar)

        with tracing():
            result_traced = a + (b + c)
        assert isinstance(result_traced.right, pybamm.Addition)

    def test_subtract_constant_move_suppressed(self):
        """var - constant -> -constant + var is suppressed."""
        x = pybamm.Parameter("x")
        c = pybamm.Scalar(5)

        result_normal = x - c
        assert isinstance(result_normal, pybamm.Addition)

        with tracing():
            result_traced = x - c
        assert isinstance(result_traced, pybamm.Subtraction)

    def test_subtract_reassociation_suppressed(self):
        """a - (b + c) -> (a - b) - c when a, b are constant."""
        a = pybamm.Scalar(10)
        b = pybamm.Scalar(3)
        c = pybamm.Parameter("c")

        result_normal = a - (b + c)
        assert isinstance(result_normal.left, pybamm.Scalar)

        with tracing():
            result_traced = a - (b + c)
        assert isinstance(result_traced.right, pybamm.Addition)

    def test_multiply_reassociation_suppressed(self):
        """a * (b * c) -> (a * b) * c when a, b are constant."""
        a = pybamm.Scalar(2)
        b = pybamm.Scalar(3)
        c = pybamm.Parameter("c")

        result_normal = a * (b * c)
        assert isinstance(result_normal.left, pybamm.Scalar)
        assert result_normal.left.evaluate() == 6

        with tracing():
            result_traced = a * (b * c)
        assert isinstance(result_traced.right, pybamm.Multiplication)

    def test_divide_constant_move_suppressed(self):
        """var / constant -> (1/constant) * var is suppressed."""
        x = pybamm.Parameter("x")
        c = pybamm.Scalar(4)

        result_normal = x / c
        assert isinstance(result_normal, pybamm.Multiplication)

        with tracing():
            result_traced = x / c
        assert isinstance(result_traced, pybamm.Division)


class TestTracingPreservesFPSemantics:
    """Verify that tracing preserves the exact floating-point results
    that the original callable would produce."""

    @pytest.mark.parametrize(
        "func",
        [
            lambda x: (2.0 * x) ** 0.5,
            lambda x: 1.1 + (2.2 + x),
            lambda x: x - 3.3,
            lambda x: 10.0 - (3.0 + x),
            lambda x: 2.5 * (3.7 * x),
            lambda x: x / 7.0,
        ],
        ids=[
            "power_distribution",
            "add_reassociation",
            "subtract_constant_move",
            "subtract_reassociation",
            "multiply_reassociation",
            "divide_constant_move",
        ],
    )
    def test_traced_expression_matches_callable(self, func):
        """The traced expression tree must evaluate identically to the
        original Python callable for arbitrary inputs."""
        test_values = np.array([0.1, 1.0, 2.5, 10.0, 100.0])

        p = pybamm.InputParameter("x")
        with tracing():
            expr = func(p)

        for val in test_values:
            expected = func(val)
            result = expr.evaluate(inputs={"x": np.array([val])})
            actual = float(np.squeeze(result))
            assert actual == expected, (
                f"FP mismatch for input {val}: {actual} != {expected}"
            )
