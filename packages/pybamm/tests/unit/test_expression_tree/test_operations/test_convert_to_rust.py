import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal


class TestRustExprGraph:
    """Tests for the Rust expression graph via PyO3 bindings."""

    def test_import(self):
        from pybamm.rust import Expr, ExprGraph  # noqa: F401

    def test_scalar(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        s = g.scalar(3.14)
        result = g.eval_to_float(s, 0.0, [], [], [])
        assert result == pytest.approx(3.14)

    def test_time(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        t = g.time()
        result = g.eval_to_float(t, 2.5, [], [], [])
        assert result == pytest.approx(2.5)

    def test_state_vector(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        sv = g.state_vector(1, 3)
        y = np.array([10.0, 20.0, 30.0, 40.0])
        result = g.eval_to_array(sv, 0.0, y, np.array([]), [])
        assert_array_almost_equal(result, [20.0, 30.0])

    def test_state_vector_dot(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        sv_dot = g.state_vector_dot(0, 2)
        y_dot = np.array([100.0, 200.0, 300.0])
        result = g.eval_to_array(sv_dot, 0.0, np.array([]), y_dot, [])
        assert_array_almost_equal(result, [100.0, 200.0])

    def test_input_parameter(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        p = g.input_parameter("C_rate")
        result = g.eval_to_float(p, 0.0, [], [], [1.5])
        assert result == pytest.approx(1.5)

    def test_array(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        a = g.array(np.array([1.0, 2.0, 3.0]))
        result = g.eval_to_array(a, 0.0, np.array([]), np.array([]), [])
        assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_add_scalars(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        a = g.scalar(2.0)
        b = g.scalar(3.0)
        c = g.add(a, b)
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(5.0)

    def test_add_dunder(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        a = g.scalar(2.0)
        b = g.scalar(3.0)
        c = a + b
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(5.0)

    def test_mul_dunder(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        a = g.scalar(4.0)
        b = g.scalar(5.0)
        c = a * b
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(20.0)

    def test_sub_dunder(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        a = g.scalar(10.0)
        b = g.scalar(3.0)
        c = a - b
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(7.0)

    def test_neg_dunder(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        a = g.scalar(5.0)
        b = -a
        result = g.eval_to_float(b, 0.0, [], [], [])
        assert result == pytest.approx(-5.0)

    def test_truediv_dunder(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        a = g.scalar(10.0)
        b = g.scalar(4.0)
        c = a / b
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(2.5)

    def test_pow_dunder(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        a = g.scalar(3.0)
        b = g.scalar(2.0)
        c = a**b
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(9.0)

    def test_nested_expression(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        two = g.scalar(2.0)
        three = g.scalar(3.0)
        four = g.scalar(4.0)
        one = g.scalar(1.0)
        result_expr = (two + three) * four - one
        result = g.eval_to_float(result_expr, 0.0, [], [], [])
        assert result == pytest.approx(19.0)

    def test_vector_add(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        a = g.array(np.array([1.0, 2.0, 3.0]))
        b = g.array(np.array([10.0, 20.0, 30.0]))
        c = a + b
        result = g.eval_to_array(c, 0.0, np.array([]), np.array([]), [])
        assert_array_almost_equal(result, [11.0, 22.0, 33.0])

    def test_scalar_times_vector(self):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        s = g.scalar(2.0)
        v = g.array(np.array([1.0, 2.0, 3.0]))
        c = s * v
        result = g.eval_to_array(c, 0.0, np.array([]), np.array([]), [])
        assert_array_almost_equal(result, [2.0, 4.0, 6.0])


class TestSymbolToRust:
    """Tests that _to_rust() conversion matches Symbol.evaluate()."""

    def test_scalar_to_rust(self):
        import pybamm
        from pybamm.rust import ExprGraph

        s = pybamm.Scalar(3.14)
        g = ExprGraph()
        rust_symbols = {}
        expr = s.to_rust(g, rust_symbols)
        result = g.eval_to_float(expr, 0.0, [], [], [])
        assert result == pytest.approx(s.evaluate())

    def test_array_to_rust(self):
        import pybamm
        from pybamm.rust import ExprGraph

        a = pybamm.Vector(np.array([1.0, 2.0, 3.0]))
        g = ExprGraph()
        rust_symbols = {}
        expr = a.to_rust(g, rust_symbols)
        result = g.eval_to_array(expr, 0.0, np.array([]), np.array([]), [])
        expected = a.evaluate()
        assert_array_almost_equal(result, expected.flatten())

    def test_addition_to_rust(self):
        import pybamm
        from pybamm.rust import ExprGraph

        a = pybamm.Scalar(2.0)
        b = pybamm.Scalar(3.0)
        expr_pybamm = a + b
        g = ExprGraph()
        rust_symbols = {}
        expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_float(expr, 0.0, [], [], [])
        assert result == pytest.approx(expr_pybamm.evaluate())

    def test_nested_arithmetic_to_rust(self):
        import pybamm
        from pybamm.rust import ExprGraph

        a = pybamm.Scalar(2.0)
        b = pybamm.Scalar(3.0)
        c = pybamm.Scalar(4.0)
        expr_pybamm = (a + b) * c - pybamm.Scalar(1.0)
        g = ExprGraph()
        rust_symbols = {}
        expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_float(expr, 0.0, [], [], [])
        assert result == pytest.approx(expr_pybamm.evaluate())

    def test_state_vector_to_rust(self):
        import pybamm
        from pybamm.rust import ExprGraph

        sv = pybamm.StateVector(slice(1, 4))
        y = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        g = ExprGraph()
        rust_symbols = {}
        expr = sv.to_rust(g, rust_symbols)
        result = g.eval_to_array(expr, 0.0, y, np.array([]), [])
        expected = sv.evaluate(y=y)
        assert_array_almost_equal(result, expected.flatten())

    def test_caching(self):
        import pybamm
        from pybamm.rust import ExprGraph

        s = pybamm.Scalar(3.14)
        g = ExprGraph()
        rust_symbols = {}
        expr1 = s.to_rust(g, rust_symbols)
        expr2 = s.to_rust(g, rust_symbols)
        assert expr1.id == expr2.id

    def test_negation_to_rust(self):
        import pybamm
        from pybamm.rust import ExprGraph

        a = pybamm.Scalar(5.0)
        expr_pybamm = -a
        g = ExprGraph()
        rust_symbols = {}
        expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_float(expr, 0.0, [], [], [])
        assert result == pytest.approx(expr_pybamm.evaluate())

    def test_abs_to_rust(self):
        import pybamm
        from pybamm.rust import ExprGraph

        a = pybamm.Scalar(-5.0)
        expr_pybamm = pybamm.AbsoluteValue(a)
        g = ExprGraph()
        rust_symbols = {}
        expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_float(expr, 0.0, [], [], [])
        assert result == pytest.approx(expr_pybamm.evaluate())


class TestSpecificFunctionToRust:
    """Tests that SpecificFunction._to_rust() conversions match Symbol.evaluate()."""

    def _eval_scalar(self, pybamm_expr, g=None):
        """Helper: convert pybamm_expr to Rust and eval as float."""
        from pybamm.rust import ExprGraph

        if g is None:
            g = ExprGraph()
        rust_symbols = {}
        expr = pybamm_expr.to_rust(g, rust_symbols)
        return g.eval_to_float(expr, 0.0, [], [], [])

    def test_sqrt_scalar(self):
        import pybamm

        expr = pybamm.sqrt(pybamm.Scalar(9.0))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_sqrt_vector(self):
        import pybamm
        from pybamm.rust import ExprGraph

        v = pybamm.Vector(np.array([1.0, 4.0, 9.0, 16.0]))
        expr_pybamm = pybamm.sqrt(v)
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        expected = expr_pybamm.evaluate().flatten()
        assert_array_almost_equal(result, expected)

    def test_exp_scalar(self):
        import pybamm

        expr = pybamm.exp(pybamm.Scalar(2.0))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_log_scalar(self):
        import pybamm

        expr = pybamm.log(pybamm.Scalar(np.e))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_sin_scalar(self):
        import pybamm

        expr = pybamm.sin(pybamm.Scalar(np.pi / 6))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_cos_scalar(self):
        import pybamm

        expr = pybamm.cos(pybamm.Scalar(np.pi / 3))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_tanh_scalar(self):
        import pybamm

        expr = pybamm.tanh(pybamm.Scalar(1.5))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_sinh_scalar(self):
        import pybamm

        expr = pybamm.sinh(pybamm.Scalar(1.0))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_cosh_scalar(self):
        import pybamm

        expr = pybamm.cosh(pybamm.Scalar(1.0))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_arcsinh_scalar(self):
        import pybamm

        expr = pybamm.arcsinh(pybamm.Scalar(2.0))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_arctan_scalar(self):
        import pybamm
        from pybamm.rust import ExprGraph

        # Use InputParameter to prevent constant folding
        p = pybamm.InputParameter("x")
        expr = pybamm.arctan(p)
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr.to_rust(g, rust_symbols)
        result = g.eval_to_float(rust_expr, 0.0, [], [], [1.0])
        expected = float(expr.evaluate(inputs={"x": 1.0}))
        assert result == pytest.approx(expected)

    def test_erf_scalar(self):
        import pybamm
        from pybamm.rust import ExprGraph

        # Use InputParameter to prevent constant folding
        p = pybamm.InputParameter("x")
        expr = pybamm.erf(p)
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr.to_rust(g, rust_symbols)
        result = g.eval_to_float(rust_expr, 0.0, [], [], [1.0])
        expected = float(expr.evaluate(inputs={"x": 1.0}))
        # erf approximation has ~1.5e-7 max error
        assert result == pytest.approx(expected, rel=1e-5)

    def test_sign_positive(self):
        import pybamm

        expr = pybamm.sign(pybamm.Scalar(3.5))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_sign_negative(self):
        import pybamm

        expr = pybamm.sign(pybamm.Scalar(-7.0))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_floor_scalar(self):
        import pybamm

        expr = pybamm.Floor(pybamm.Scalar(3.7))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_ceiling_scalar(self):
        import pybamm

        expr = pybamm.Ceiling(pybamm.Scalar(3.2))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    def test_nested_function(self):
        """Test nested: exp(sqrt(x)) for x=4.0 should give e^2."""
        import pybamm

        expr = pybamm.exp(pybamm.sqrt(pybamm.Scalar(4.0)))
        result = self._eval_scalar(expr)
        assert result == pytest.approx(float(expr.evaluate()))

    @pytest.mark.parametrize(
        "x_val,exponent",
        [
            (4.0, 0.5),
            (-4.0, 0.5),
            (1.5, 1.5),
            (1e-12, 0.5),  # near zero, regularisation regime dominates
            (0.0, 0.5),  # exactly zero
            # One case per chain in `_positive_base_pow_chain`, which keys off the
            # inner exponent (a-1)/2 in {-0.5, -0.25, 0.25, 0.5, 1}, plus a
            # non-chain exponent that falls back to runtime pow.
            (4.0, 0.0),
            (-2.0, 0.0),
            (4.0, 1.5),
            (4.0, 2.0),
            (-3.0, 2.0),
            (4.0, 3.0),
            (4.0, 0.7),
            (-4.0, 0.7),
        ],
    )
    def test_reg_power_scalar(self, x_val, exponent):
        """RegPower(x, a) on the Rust path matches the numpy evaluator."""
        import pybamm
        from pybamm.rust import ExprGraph

        p = pybamm.InputParameter("x")
        expr = pybamm.reg_power(p, exponent)
        g = ExprGraph()
        rust_expr = expr.to_rust(g, {})
        result = g.eval_to_float(rust_expr, 0.0, [], [], [x_val])
        expected = float(expr.evaluate(inputs={"x": x_val}))
        assert result == pytest.approx(expected, rel=1e-12, abs=1e-15)

    def test_reg_power_with_scale(self):
        """RegPower with explicit non-unit scale matches the numpy evaluator."""
        import pybamm
        from pybamm.rust import ExprGraph

        p = pybamm.InputParameter("x")
        expr = pybamm.reg_power(p, 0.7, scale=2.5)
        g = ExprGraph()
        rust_expr = expr.to_rust(g, {})
        for x_val in (-3.0, -0.1, 1e-9, 0.0, 0.4, 5.0):
            result = g.eval_to_float(rust_expr, 0.0, [], [], [x_val])
            expected = float(expr.evaluate(inputs={"x": x_val}))
            assert result == pytest.approx(expected, rel=1e-12, abs=1e-14)


class TestIndexToRust:
    """Tests for Index (slicing) node conversion to Rust."""

    def test_index_vector_slice(self):
        """Index a Vector with slice(1, 3) → elements at positions 1 and 2."""
        import pybamm
        from pybamm.rust import ExprGraph

        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr_pybamm = pybamm.Index(v, slice(1, 3))
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        expected = expr_pybamm.evaluate().flatten()
        assert_array_almost_equal(result, expected)

    def test_index_state_vector_slice(self):
        """Index a StateVector with slice(2, 4) using specific y values."""
        import pybamm
        from pybamm.rust import ExprGraph

        sv = pybamm.StateVector(slice(0, 6))
        expr_pybamm = pybamm.Index(sv, slice(2, 4))
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_array(rust_expr, 0.0, y, np.array([]), [])
        expected = expr_pybamm.evaluate(y=y.reshape(-1, 1)).flatten()
        assert_array_almost_equal(result, expected)

    def test_index_negative_last_element(self):
        """Index(vector, -1) → last element, via slice(-1, None)."""
        import pybamm
        from pybamm.rust import ExprGraph

        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr_pybamm = pybamm.Index(v, -1)
        g = ExprGraph()
        rust_expr = expr_pybamm.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        expected = expr_pybamm.evaluate().flatten()  # [50.0]
        assert_array_almost_equal(result, expected)

    def test_index_step_raises(self):
        """A strided Index (step != 1) must raise, not silently return a
        contiguous slice."""
        import pybamm
        from pybamm.rust import ExprGraph

        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr_pybamm = pybamm.Index(v, slice(0, 4, 2))
        g = ExprGraph()
        with pytest.raises(NotImplementedError, match="step"):
            expr_pybamm.to_rust(g, {})

    def test_index_over_negative_start_clamps(self):
        """A start more negative than -size clamps to 0 (numpy semantics)."""
        import pybamm
        from pybamm.rust import ExprGraph

        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr_pybamm = pybamm.Index(v, slice(-7, 5))
        g = ExprGraph()
        rust_expr = expr_pybamm.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        expected = expr_pybamm.evaluate().flatten()  # full [10,20,30,40,50]
        assert_array_almost_equal(result, expected)


class TestNewBinaryOperatorsToRust:
    """Tests for new binary operators: Minimum, Maximum, Modulo, Hypot,
    EqualHeaviside, NotEqualHeaviside, Equality, and sparse MatMul."""

    def _eval_scalar(self, pybamm_expr, inputs=None):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        rust_symbols = {}
        expr = pybamm_expr.to_rust(g, rust_symbols)
        inputs_list = list(inputs.values()) if inputs else []
        return g.eval_to_float(expr, 0.0, [], [], inputs_list)

    def _eval_array(self, pybamm_expr, y=None, inputs=None):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        rust_symbols = {}
        expr = pybamm_expr.to_rust(g, rust_symbols)
        y_arr = np.array(y) if y is not None else np.array([])
        inputs_list = list(inputs.values()) if inputs else []
        return g.eval_to_array(expr, 0.0, y_arr, np.array([]), inputs_list)

    def test_minimum_scalars(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.minimum(a, b)
        result = self._eval_scalar(expr, inputs={"a": 3.0, "b": 5.0})
        expected = float(expr.evaluate(inputs={"a": 3.0, "b": 5.0}))
        assert result == pytest.approx(expected)

    def test_minimum_reversed(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.minimum(a, b)
        result = self._eval_scalar(expr, inputs={"a": 7.0, "b": 2.0})
        expected = float(expr.evaluate(inputs={"a": 7.0, "b": 2.0}))
        assert result == pytest.approx(expected)

    def test_maximum_scalars(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.maximum(a, b)
        result = self._eval_scalar(expr, inputs={"a": 3.0, "b": 5.0})
        expected = float(expr.evaluate(inputs={"a": 3.0, "b": 5.0}))
        assert result == pytest.approx(expected)

    def test_maximum_reversed(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.maximum(a, b)
        result = self._eval_scalar(expr, inputs={"a": 7.0, "b": 2.0})
        expected = float(expr.evaluate(inputs={"a": 7.0, "b": 2.0}))
        assert result == pytest.approx(expected)

    def test_modulo_scalar(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.Modulo(a, b)
        result = self._eval_scalar(expr, inputs={"a": 10.0, "b": 3.0})
        expected = float(expr.evaluate(inputs={"a": 10.0, "b": 3.0}))
        assert result == pytest.approx(expected)

    def test_hypot_scalar(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.hypot(a, b)
        result = self._eval_scalar(expr, inputs={"a": 3.0, "b": 4.0})
        expected = float(expr.evaluate(inputs={"a": 3.0, "b": 4.0}))
        assert result == pytest.approx(expected)

    def test_equal_heaviside_true(self):
        """3.0 <= 5.0 should return 1."""
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.EqualHeaviside(a, b)
        result = self._eval_scalar(expr, inputs={"a": 3.0, "b": 5.0})
        assert result == pytest.approx(1.0)

    def test_equal_heaviside_equal(self):
        """5.0 <= 5.0 should return 1 (equal case)."""
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.EqualHeaviside(a, b)
        result = self._eval_scalar(expr, inputs={"a": 5.0, "b": 5.0})
        assert result == pytest.approx(1.0)

    def test_equal_heaviside_false(self):
        """7.0 <= 5.0 should return 0."""
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.EqualHeaviside(a, b)
        result = self._eval_scalar(expr, inputs={"a": 7.0, "b": 5.0})
        assert result == pytest.approx(0.0)

    def test_not_equal_heaviside_true(self):
        """3.0 < 5.0 should return 1."""
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.NotEqualHeaviside(a, b)
        result = self._eval_scalar(expr, inputs={"a": 3.0, "b": 5.0})
        assert result == pytest.approx(1.0)

    def test_not_equal_heaviside_equal(self):
        """5.0 < 5.0 should return 0 (strict inequality)."""
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.NotEqualHeaviside(a, b)
        result = self._eval_scalar(expr, inputs={"a": 5.0, "b": 5.0})
        assert result == pytest.approx(0.0)

    def test_equality_equal(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.Equality(a, b)
        result = self._eval_scalar(expr, inputs={"a": 3.0, "b": 3.0})
        assert result == pytest.approx(1.0)

    def test_equality_not_equal(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.Equality(a, b)
        result = self._eval_scalar(expr, inputs={"a": 3.0, "b": 4.0})
        assert result == pytest.approx(0.0)

    def test_sparse_matrix_matmul(self):
        """Test sparse Matrix @ StateVector conversion to Rust."""
        from scipy.sparse import csr_matrix as scipy_csr

        import pybamm
        from pybamm.rust import ExprGraph

        # 2x3 sparse matrix: [1 0 0; 0 0 2]
        data = np.array([1.0, 2.0])
        row = np.array([0, 1])
        col = np.array([0, 2])
        sparse_mat = scipy_csr((data, (row, col)), shape=(2, 3))

        mat = pybamm.Matrix(sparse_mat)
        sv = pybamm.StateVector(slice(0, 3))
        expr = mat @ sv

        y = np.array([10.0, 20.0, 30.0])
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr.to_rust(g, rust_symbols)
        result = g.eval_to_array(rust_expr, 0.0, y, np.array([]), [])
        expected = expr.evaluate(y=y.reshape(-1, 1)).flatten()
        assert_array_almost_equal(result, expected)


class TestInterpolantToRust:
    """Tests for Interpolant._to_rust() — 1D linear, cubic, and pchip interpolation."""

    def test_1d_linear(self):
        """1D linear interpolant: y=2x on [0,1], evaluate at x=0.4 → ~0.8."""
        import pybamm
        from pybamm.rust import ExprGraph

        x = np.linspace(0, 1, 50)
        y = 2 * x
        sv = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(x, y, sv)

        y_test = np.array([0.4])
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = interp.to_rust(g, rust_symbols)
        result = g.eval_to_array(rust_expr, 0.0, y_test, np.array([]), [])
        expected = interp.evaluate(y=y_test.reshape(-1, 1)).flatten()
        assert_array_almost_equal(result, expected, decimal=5)

    def test_1d_linear_multiple_values(self):
        """1D linear interpolant: y=2x on [0,1], evaluate at [0.4, 0.6] → [0.8, 1.2]."""
        import pybamm
        from pybamm.rust import ExprGraph

        x = np.linspace(0, 1, 50)
        y = 2 * x
        sv = pybamm.StateVector(slice(0, 2))
        interp = pybamm.Interpolant(x, y, sv)

        y_test = np.array([0.4, 0.6])
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = interp.to_rust(g, rust_symbols)
        result = g.eval_to_array(rust_expr, 0.0, y_test, np.array([]), [])
        expected = interp.evaluate(y=y_test.reshape(-1, 1)).flatten()
        assert_array_almost_equal(result, expected, decimal=5)

    def test_1d_linear_extrapolation_matches_evaluate(self):
        """Out-of-domain linear interp extends, matching Symbol.evaluate()."""
        import pybamm
        from pybamm.rust import ExprGraph

        x = np.linspace(0.0, 1.0, 11)
        y = 3.0 * x + 1.0  # slope 3
        sv = pybamm.StateVector(slice(0, 2))
        interp = pybamm.Interpolant(x, y, sv)  # default linear, extrapolate=True

        y_test = np.array([-0.5, 1.5])  # both out of [0, 1]
        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, y_test, np.array([]), [])
        expected = interp.evaluate(y=y_test.reshape(-1, 1)).flatten()
        assert_array_almost_equal(result, expected, decimal=10)

    def _check_interp_parity(self, interpolator):
        import pybamm
        from pybamm.rust import ExprGraph

        # Non-uniform grid so coefficients are non-trivial.
        x = np.array([0.0, 0.5, 1.0, 2.0, 3.5, 5.0])
        y = np.array([0.0, 0.4, 1.1, 3.9, 12.0, 24.0])
        sv = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(x, y, sv, interpolator=interpolator)

        g = ExprGraph()
        rust_symbols = {}
        rust_expr = interp.to_rust(g, rust_symbols)
        # In-domain, at breakpoints, and out-of-domain (extend) on both sides.
        for q in [-1.0, 0.0, 0.5, 1.3, 2.0, 3.5, 5.0, 7.5]:
            yq = np.array([q])
            result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
            expected = interp.evaluate(y=yq.reshape(-1, 1)).flatten()
            assert_array_almost_equal(result, expected, decimal=10)

    def test_1d_cubic_matches_evaluate(self):
        self._check_interp_parity("cubic")

    def test_1d_pchip_matches_evaluate(self):
        self._check_interp_parity("pchip")

    def test_1d_cubic_vector_child(self):
        """Element-wise over a vector child matches evaluate()."""
        import pybamm
        from pybamm.rust import ExprGraph

        x = np.linspace(0.0, 4.0, 20)
        y = np.sin(x)
        sv = pybamm.StateVector(slice(0, 3))
        interp = pybamm.Interpolant(x, y, sv, interpolator="cubic")

        y_test = np.array([0.7, 2.1, 5.0])  # last is out-of-domain (extend)
        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, y_test, np.array([]), [])
        expected = interp.evaluate(y=y_test.reshape(-1, 1)).flatten()
        assert_array_almost_equal(result, expected, decimal=10)

    @pytest.mark.parametrize("interpolator", ["linear", "cubic", "pchip"])
    def test_1d_extrapolate_false_extends(self, interpolator):
        """extrapolate=False is intentionally ignored on the Rust path (parity
        spec Decision 3): in-domain matches evaluate(), out-of-domain extends
        the boundary polynomial instead of returning NaN (a NaN would poison
        the solver residual; the domain is guarded by extrapolation events)."""
        import pybamm
        from pybamm.rust import ExprGraph

        x = np.array([0.0, 0.5, 1.0, 2.0, 3.5, 5.0])
        y = np.array([0.0, 0.4, 1.1, 3.9, 12.0, 24.0])
        sv = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(
            x, y, sv, interpolator=interpolator, extrapolate=False
        )
        extend = pybamm.Interpolant(x, y, sv, interpolator=interpolator)

        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        # In-domain: parity with evaluate().
        yq = np.array([1.3])
        result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
        expected = interp.evaluate(y=yq.reshape(-1, 1)).flatten()
        assert_array_almost_equal(result, expected, decimal=10)
        # Out-of-domain: evaluate() gives NaN, Rust extends (extrapolate=True).
        for q in [-1.0, 7.5]:
            yq = np.array([q])
            assert np.isnan(interp.evaluate(y=yq.reshape(-1, 1))).all()
            result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
            expected = extend.evaluate(y=yq.reshape(-1, 1)).flatten()
            assert_array_almost_equal(result, expected, decimal=10)

    @pytest.mark.parametrize("interpolator", ["linear", "cubic", "pchip"])
    def test_1d_vector_valued_y_matches_evaluate(self, interpolator):
        """Vector-valued y (y.ndim == 2) stacks one interpolant per column in
        evaluate()'s column order."""
        import pybamm
        from pybamm.rust import ExprGraph

        # Non-uniform grid and distinct nonlinear columns so a column swap or
        # wrong-length table cannot cancel out.
        x = np.array([0.0, 0.5, 1.0, 2.0, 3.5, 5.0])
        y = np.column_stack([np.sin(x), np.cos(x), x**2])
        sv = pybamm.StateVector(slice(0, 1))  # scalar child (constructor requires)
        interp = pybamm.Interpolant(x, y, sv, interpolator=interpolator)

        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        # In-domain, at breakpoints, and out-of-domain (extend) on both sides.
        for q in [-1.0, 0.0, 0.5, 1.3, 2.0, 3.5, 5.0, 7.5]:
            yq = np.array([q])
            result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
            expected = interp.evaluate(y=yq.reshape(-1, 1)).flatten()
            assert_array_almost_equal(result, expected, decimal=10)

    @pytest.mark.parametrize("interpolator", ["linear", "cubic", "pchip"])
    def test_1d_vector_valued_y_jacobian_matches_casadi(self, interpolator):
        """d/dy of the stacked columns matches the CasADi conversion."""
        import casadi

        import pybamm
        from pybamm.rust import ExprGraph

        x = np.array([0.0, 0.5, 1.0, 2.0, 3.5, 5.0])
        y = np.column_stack([np.sin(x), np.cos(x), x**2])
        sv = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(x, y, sv, interpolator=interpolator)

        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        f = g.compile(rust_expr, n_states=1)
        yq = np.array([1.3])  # strictly inside a segment (linear has knot kinks)
        rust_jac = f.jacobian("y")(0.0, yq, np.array([])).toarray()

        ys = casadi.MX.sym("y", 1)
        casadi_jac = casadi.Function(
            "J", [ys], [casadi.jacobian(interp.to_casadi(y=ys), ys)]
        )
        np.testing.assert_allclose(
            rust_jac, np.array(casadi_jac(yq)), rtol=1e-8, atol=1e-10
        )

    def test_1d_y_with_more_than_two_dims_raises(self):
        """y.ndim > 2 has no defined column order, so refuse loudly."""
        import pybamm
        from pybamm.rust import ExprGraph

        x = np.linspace(0.0, 1.0, 5)
        y = np.ones((5, 2, 2))
        sv = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(x, y, sv)
        with pytest.raises(NotImplementedError, match=r"more than two dimensions"):
            interp.to_rust(ExprGraph(), {})


class TestInterpolantNDToRust:
    """2D/3D interpolants lower to the Rust ND tensor-product node and match
    Symbol.evaluate() (RegularGridInterpolator) in-domain, at every grid
    point, and out-of-domain (extend), including all-axes-out corners."""

    # Non-uniform grids so coefficients and cell strides are non-trivial.
    x0 = np.array([0.0, 0.5, 1.3, 2.0, 3.1, 4.0])
    x1 = np.array([-1.0, -0.3, 0.4, 1.0, 2.2])
    x2 = np.array([0.0, 1.0, 2.5, 3.5, 5.0])

    @staticmethod
    def _pts(axes):
        rng = np.random.default_rng(7)
        ins = np.column_stack([rng.uniform(a[0], a[-1], 12) for a in axes])
        grid = np.array(list(itertools.product(*axes)))
        mid = [0.5 * (a[0] + a[-1]) for a in axes]
        outs = []
        for i, a in enumerate(axes):
            lo = list(mid)
            lo[i] = a[0] - 0.7
            outs.append(lo)
            hi = list(mid)
            hi[i] = a[-1] + 0.9
            outs.append(hi)
        outs.append([a[0] - 0.8 for a in axes])  # all-axes-out corner
        outs.append([a[-1] + 1.1 for a in axes])  # all-axes-out corner
        return np.vstack([ins, grid, np.asarray(outs)])

    def _check_parity(self, x, y, interpolator):
        import pybamm
        from pybamm.rust import ExprGraph

        ndim = len(x)
        svs = tuple(pybamm.StateVector(slice(i, i + 1)) for i in range(ndim))
        interp = pybamm.Interpolant(x, y, svs, interpolator=interpolator)
        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        for p in self._pts(list(x)):
            yq = np.asarray(p, dtype=float)
            result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
            expected = np.asarray(interp.evaluate(y=yq.reshape(-1, 1))).flatten()
            assert_array_almost_equal(result, expected, decimal=10)

    @pytest.mark.parametrize("interpolator", ["linear", "cubic"])
    def test_2d_matches_evaluate(self, interpolator):
        X0, X1 = np.meshgrid(self.x0, self.x1, indexing="ij")
        y = np.sin(X0) * np.exp(-0.3 * X1) + 0.1 * X0 * X1
        self._check_parity((self.x0, self.x1), y, interpolator)

    @pytest.mark.parametrize("interpolator", ["linear", "cubic"])
    def test_3d_matches_evaluate(self, interpolator):
        X0, X1, X2 = np.meshgrid(self.x0, self.x1, self.x2, indexing="ij")
        y = np.cos(X0) * X1 + 0.5 * np.sqrt(1 + X2) + 0.05 * X0 * X1 * X2
        self._check_parity((self.x0, self.x1, self.x2), y, interpolator)

    @pytest.mark.parametrize("interpolator", ["linear", "cubic"])
    def test_2d_vector_children(self, interpolator):
        """Element-wise over equal-length vector children matches evaluate()."""
        import pybamm
        from pybamm.rust import ExprGraph

        X0, X1 = np.meshgrid(self.x0, self.x1, indexing="ij")
        y = X0**2 + X1
        sv0 = pybamm.StateVector(slice(0, 3))
        sv1 = pybamm.StateVector(slice(3, 6))
        interp = pybamm.Interpolant(
            (self.x0, self.x1), y, (sv0, sv1), interpolator=interpolator
        )
        # Last pair is out-of-domain on both axes (extend).
        y_test = np.array([0.7, 2.1, 5.0, 0.0, 1.5, -1.4])
        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, y_test, np.array([]), [])
        expected = np.asarray(interp.evaluate(y=y_test.reshape(-1, 1))).flatten()
        assert_array_almost_equal(result, expected, decimal=10)

    @pytest.mark.parametrize("interpolator", ["linear", "cubic"])
    def test_2d_extrapolate_false_extends(self, interpolator):
        """extrapolate=False is intentionally ignored on the Rust path (parity
        spec Decision 3): in-domain matches evaluate(), out-of-domain extends
        instead of returning NaN."""
        import pybamm
        from pybamm.rust import ExprGraph

        X0, X1 = np.meshgrid(self.x0, self.x1, indexing="ij")
        y = np.sin(X0) * np.exp(-0.3 * X1) + 0.1 * X0 * X1
        svs = (pybamm.StateVector(slice(0, 1)), pybamm.StateVector(slice(1, 2)))
        interp = pybamm.Interpolant(
            (self.x0, self.x1), y, svs, interpolator=interpolator, extrapolate=False
        )
        extend = pybamm.Interpolant(
            (self.x0, self.x1), y, svs, interpolator=interpolator
        )

        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        # In-domain: parity with evaluate().
        yq = np.array([1.3, 0.5])
        result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
        expected = np.asarray(interp.evaluate(y=yq.reshape(-1, 1))).flatten()
        assert_array_almost_equal(result, expected, decimal=10)
        # Out-of-domain: evaluate() gives NaN, Rust extends (extrapolate=True).
        for p in [[-0.7, 0.5], [4.9, 3.1]]:
            yq = np.asarray(p)
            assert np.isnan(interp.evaluate(y=yq.reshape(-1, 1))).all()
            result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
            expected = np.asarray(extend.evaluate(y=yq.reshape(-1, 1))).flatten()
            assert_array_almost_equal(result, expected, decimal=10)


class TestReductionToRust:
    """Tests for MaxReduce and MinReduce node conversion to Rust."""

    def test_max_reduce(self):
        """pybamm.max(Vector([1, 5, 3])) should reduce to 5.0."""
        import pybamm
        from pybamm.rust import ExprGraph

        v = pybamm.Vector(np.array([1.0, 5.0, 3.0]))
        expr_pybamm = pybamm.max(v)
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_float(rust_expr, 0.0, [], [], [])
        assert result == pytest.approx(5.0)

    def test_min_reduce(self):
        """pybamm.min(Vector([1, 5, 3])) should reduce to 1.0."""
        import pybamm
        from pybamm.rust import ExprGraph

        v = pybamm.Vector(np.array([1.0, 5.0, 3.0]))
        expr_pybamm = pybamm.min(v)
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_float(rust_expr, 0.0, [], [], [])
        assert result == pytest.approx(1.0)


class TestConcatenationToRust:
    """Tests for concatenation node conversion to Rust."""

    def test_numpy_concatenation(self):
        """NumpyConcatenation of Vector([1,2]) and Vector([3,4,5]) → [1,2,3,4,5]."""
        import pybamm
        from pybamm.rust import ExprGraph

        v1 = pybamm.Vector(np.array([1.0, 2.0]))
        v2 = pybamm.Vector(np.array([3.0, 4.0, 5.0]))
        expr_pybamm = pybamm.NumpyConcatenation(v1, v2)
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        expected = expr_pybamm.evaluate().flatten()
        assert_array_almost_equal(result, expected)

    def test_state_vector_concatenation(self):
        """NumpyConcatenation of StateVector(0:3) and StateVector(3:5), y=[1,2,3,4,5]."""
        import pybamm
        from pybamm.rust import ExprGraph

        sv1 = pybamm.StateVector(slice(0, 3))
        sv2 = pybamm.StateVector(slice(3, 5))
        expr_pybamm = pybamm.NumpyConcatenation(sv1, sv2)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_array(rust_expr, 0.0, y, np.array([]), [])
        expected = expr_pybamm.evaluate(y=y.reshape(-1, 1)).flatten()
        assert_array_almost_equal(result, expected)


class TestVectorFieldToRust:
    """Tests for vector field conversion to Rust, which stacks its components."""

    def test_vector_field(self):
        """VectorField of Vector([1,2]) and Vector([3,4]) → [1,2,3,4]."""
        import pybamm
        from pybamm.rust import ExprGraph

        v1 = pybamm.Vector(np.array([1.0, 2.0]))
        v2 = pybamm.Vector(np.array([3.0, 4.0]))
        expr_pybamm = pybamm.VectorField(v1, v2)
        g = ExprGraph()
        rust_expr = expr_pybamm.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_vector_field_matches_casadi(self):
        """Three-component state-dependent field matches the CasADi conversion."""
        import casadi

        import pybamm
        from pybamm.rust import ExprGraph

        components = [
            pybamm.StateVector(slice(0, 2)),
            2 * pybamm.StateVector(slice(2, 4)),
            pybamm.StateVector(slice(0, 2)) + pybamm.StateVector(slice(2, 4)),
        ]
        expr_pybamm = pybamm.VectorField(*components)
        y = np.array([1.0, 2.0, 3.0, 4.0])

        g = ExprGraph()
        rust_expr = expr_pybamm.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, y, np.array([]), [])

        t_sym = casadi.MX.sym("t")
        y_sym = casadi.MX.sym("y", 4)
        y_dot_sym = casadi.MX.sym("y_dot", 4)
        casadi_func = casadi.Function(
            "f", [t_sym, y_sym], [expr_pybamm.to_casadi(t_sym, y_sym, y_dot_sym, {})]
        )
        expected = np.array(casadi_func(0.0, y)).flatten()
        assert_array_almost_equal(result, expected)


class TestRustMatchesPyBaMM:
    """Cross-validation: _to_rust().eval() matches symbol.evaluate() for all supported node types."""

    def _assert_rust_eval_matches(self, expr, t=0.0, y=None, y_dot=None, inputs=None):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr.to_rust(g, rust_symbols)

        # Always flatten y/y_dot to 1D float64 for the Rust bindings
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
        inputs_list = inputs if inputs is not None else []
        pybamm_result = expr.evaluate(t=t, y=y, y_dot=y_dot)

        if np.isscalar(pybamm_result) or (
            hasattr(pybamm_result, "shape") and pybamm_result.size == 1
        ):
            result = g.eval_to_float(
                rust_expr,
                t,
                y_np.tolist(),
                y_dot_np.tolist(),
                inputs_list,
            )
            assert result == pytest.approx(float(pybamm_result), rel=1e-5, abs=1e-10)
        else:
            result = g.eval_to_array(rust_expr, t, y_np, y_dot_np, inputs_list)
            np.testing.assert_allclose(
                result, np.asarray(pybamm_result).flatten(), rtol=1e-5, atol=1e-10
            )

    def test_scalar_arithmetic(self):
        """Test +, -, *, /, **, neg, abs on scalars."""
        import pybamm

        a = pybamm.Scalar(6.0)
        b = pybamm.Scalar(4.0)

        self._assert_rust_eval_matches(a + b)
        self._assert_rust_eval_matches(a - b)
        self._assert_rust_eval_matches(a * b)
        self._assert_rust_eval_matches(a / b)
        self._assert_rust_eval_matches(a**b)
        self._assert_rust_eval_matches(-a)
        self._assert_rust_eval_matches(pybamm.AbsoluteValue(pybamm.Scalar(-7.5)))

    def test_special_functions(self):
        """Test sqrt, exp, log, sin, cos, tanh, sinh, cosh, arcsinh, erf."""
        import pybamm

        # Use InputParameter to prevent constant folding where needed
        p = pybamm.InputParameter("x")

        self._assert_rust_eval_matches(pybamm.sqrt(pybamm.Scalar(9.0)))
        self._assert_rust_eval_matches(pybamm.exp(pybamm.Scalar(2.0)))
        self._assert_rust_eval_matches(pybamm.log(pybamm.Scalar(np.e)))
        self._assert_rust_eval_matches(pybamm.sin(pybamm.Scalar(np.pi / 6)))
        self._assert_rust_eval_matches(pybamm.cos(pybamm.Scalar(np.pi / 3)))
        self._assert_rust_eval_matches(pybamm.tanh(pybamm.Scalar(1.5)))
        self._assert_rust_eval_matches(pybamm.sinh(pybamm.Scalar(1.0)))
        self._assert_rust_eval_matches(pybamm.cosh(pybamm.Scalar(1.0)))
        self._assert_rust_eval_matches(pybamm.arcsinh(pybamm.Scalar(2.0)))

        # erf via InputParameter (approximation has ~1.5e-7 max error, rel=1e-5 handles it)
        from pybamm.rust import ExprGraph

        expr_erf = pybamm.erf(p)
        g = ExprGraph()
        rust_symbols = {}
        re = expr_erf.to_rust(g, rust_symbols)
        rust_result = g.eval_to_float(re, 0.0, [], [], [1.0])
        pybamm_result = float(expr_erf.evaluate(inputs={"x": 1.0}))
        assert rust_result == pytest.approx(pybamm_result, rel=1e-5)

    def test_floor_ceiling(self):
        """Test Floor and Ceiling on 3.3."""
        import pybamm

        # Use InputParameter to avoid constant folding
        p = pybamm.InputParameter("x")
        expr_floor = pybamm.Floor(p)
        expr_ceil = pybamm.Ceiling(p)

        from pybamm.rust import ExprGraph

        for expr in (expr_floor, expr_ceil):
            g = ExprGraph()
            rust_symbols = {}
            re = expr.to_rust(g, rust_symbols)
            rust_result = g.eval_to_float(re, 0.0, [], [], [3.3])
            pybamm_result = float(expr.evaluate(inputs={"x": 3.3}))
            assert rust_result == pytest.approx(pybamm_result)

    def test_sign(self):
        """Test sign on positive and negative values."""
        import pybamm

        # Use InputParameter to prevent constant folding
        p = pybamm.InputParameter("x")
        expr = pybamm.sign(p)

        from pybamm.rust import ExprGraph

        # Positive value: sign(5.0) == 1
        g = ExprGraph()
        rust_symbols = {}
        re = expr.to_rust(g, rust_symbols)
        rust_result = g.eval_to_float(re, 0.0, [], [], [5.0])
        pybamm_result = float(expr.evaluate(inputs={"x": 5.0}))
        assert rust_result == pytest.approx(pybamm_result)

        # Negative value: sign(-3.0) == -1
        g2 = ExprGraph()
        rust_symbols2 = {}
        re2 = expr.to_rust(g2, rust_symbols2)
        rust_result2 = g2.eval_to_float(re2, 0.0, [], [], [-3.0])
        pybamm_result2 = float(expr.evaluate(inputs={"x": -3.0}))
        assert rust_result2 == pytest.approx(pybamm_result2)

        # Zero: sign(0.0) == 0.0 (matches numpy convention)
        g3 = ExprGraph()
        rust_symbols3 = {}
        re3 = expr.to_rust(g3, rust_symbols3)
        rust_zero = g3.eval_to_float(re3, 0.0, [], [], [0.0])
        pybamm_zero = float(expr.evaluate(inputs={"x": 0.0}))
        assert rust_zero == pytest.approx(pybamm_zero)

    def test_modulo_minimum_maximum(self):
        """Test Modulo(7,3), Minimum(7,3), Maximum(7,3)."""
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")

        from pybamm.rust import ExprGraph

        for expr_factory, expected in [
            (pybamm.Modulo, 1.0),
            (pybamm.minimum, 3.0),
            (pybamm.maximum, 7.0),
        ]:
            expr = expr_factory(a, b)
            g = ExprGraph()
            rust_symbols = {}
            re = expr.to_rust(g, rust_symbols)
            rust_result = g.eval_to_float(re, 0.0, [], [], [7.0, 3.0])
            pybamm_result = float(expr.evaluate(inputs={"a": 7.0, "b": 3.0}))
            assert rust_result == pytest.approx(pybamm_result)
            assert rust_result == pytest.approx(expected)

    def test_hypot(self):
        """Test Hypot(3,4) == 5."""
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.hypot(a, b)

        from pybamm.rust import ExprGraph

        g = ExprGraph()
        rust_symbols = {}
        re = expr.to_rust(g, rust_symbols)
        rust_result = g.eval_to_float(re, 0.0, [], [], [3.0, 4.0])
        pybamm_result = float(expr.evaluate(inputs={"a": 3.0, "b": 4.0}))
        assert rust_result == pytest.approx(pybamm_result)
        assert rust_result == pytest.approx(5.0)

    def test_state_vector_arithmetic(self):
        """2*sv + 1 evaluated at y=[1,2,3]."""
        import pybamm

        sv = pybamm.StateVector(slice(0, 3))
        expr = pybamm.Scalar(2.0) * sv + pybamm.Scalar(1.0)
        y = np.array([1.0, 2.0, 3.0])
        self._assert_rust_eval_matches(expr, y=y.reshape(-1, 1))

    def test_multi_slice_state_vector_and_dot(self):
        """Multi-slice StateVector/StateVectorDot lower as concat of slices."""
        import pybamm

        sv = pybamm.StateVector(slice(0, 2), slice(4, 6))
        y = np.arange(6.0)
        self._assert_rust_eval_matches(sv, y=y.reshape(-1, 1))

        sv_dot = pybamm.StateVectorDot(slice(0, 2), slice(4, 6))
        y_dot = np.arange(10.0, 16.0)
        self._assert_rust_eval_matches(sv_dot, y_dot=y_dot.reshape(-1, 1))

    def test_heaviside(self):
        """Test EqualHeaviside(2,3) and NotEqualHeaviside(3,3)."""
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")

        from pybamm.rust import ExprGraph

        # EqualHeaviside(2, 3): 2 <= 3 → 1 (True)
        expr_eh = pybamm.EqualHeaviside(a, b)
        g = ExprGraph()
        rust_symbols = {}
        re = expr_eh.to_rust(g, rust_symbols)
        rust_result = g.eval_to_float(re, 0.0, [], [], [2.0, 3.0])
        pybamm_result = float(expr_eh.evaluate(inputs={"a": 2.0, "b": 3.0}))
        assert rust_result == pytest.approx(pybamm_result)
        assert rust_result == pytest.approx(1.0)

        # NotEqualHeaviside(3, 3): 3 < 3 → 0 (False, strict inequality)
        expr_neh = pybamm.NotEqualHeaviside(a, b)
        g2 = ExprGraph()
        rust_symbols2 = {}
        re2 = expr_neh.to_rust(g2, rust_symbols2)
        rust_result2 = g2.eval_to_float(re2, 0.0, [], [], [3.0, 3.0])
        pybamm_result2 = float(expr_neh.evaluate(inputs={"a": 3.0, "b": 3.0}))
        assert rust_result2 == pytest.approx(pybamm_result2)
        assert rust_result2 == pytest.approx(0.0)

    def test_interpolation_1d_linear(self):
        """Interpolant with y=2x, evaluate at [0.4, 0.6]."""
        import pybamm

        x = np.linspace(0, 1, 50)
        y_data = 2 * x
        sv = pybamm.StateVector(slice(0, 2))
        interp = pybamm.Interpolant(x, y_data, sv)

        y_test = np.array([0.4, 0.6])
        self._assert_rust_eval_matches(interp, y=y_test.reshape(-1, 1))

    def test_max_min_reduction(self):
        """pybamm.max and pybamm.min on arrays."""
        import pybamm

        v = pybamm.Vector(np.array([1.0, 5.0, 3.0]))
        self._assert_rust_eval_matches(pybamm.max(v))
        self._assert_rust_eval_matches(pybamm.min(v))

    def test_index_slice(self):
        """Index(Vector, slice(1,3)) → elements at positions 1 and 2."""
        import pybamm

        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr = pybamm.Index(v, slice(1, 3))
        self._assert_rust_eval_matches(expr)

    def test_concatenation(self):
        """NumpyConcatenation of two vectors."""
        import pybamm

        v1 = pybamm.Vector(np.array([1.0, 2.0]))
        v2 = pybamm.Vector(np.array([3.0, 4.0, 5.0]))
        expr = pybamm.NumpyConcatenation(v1, v2)
        self._assert_rust_eval_matches(expr)

    def test_sparse_matmul(self):
        """Sparse Matrix @ StateVector."""
        from scipy.sparse import csr_matrix as scipy_csr

        import pybamm

        data = np.array([1.0, 2.0])
        row = np.array([0, 1])
        col = np.array([0, 2])
        sparse_mat = scipy_csr((data, (row, col)), shape=(2, 3))
        mat = pybamm.Matrix(sparse_mat)
        sv = pybamm.StateVector(slice(0, 3))
        expr = mat @ sv

        y = np.array([10.0, 20.0, 30.0])
        self._assert_rust_eval_matches(expr, y=y.reshape(-1, 1))

    def test_composed_expression(self):
        """Realistic composed expression: sqrt(sv**2 + 1) * exp(-t) with StateVector and Time."""
        import pybamm

        sv = pybamm.StateVector(slice(0, 1))
        t_sym = pybamm.t
        expr = pybamm.sqrt(sv**2 + pybamm.Scalar(1.0)) * pybamm.exp(-t_sym)

        y = np.array([3.0])
        # sqrt(9 + 1) * exp(-2) ≈ 0.4280
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr.to_rust(g, rust_symbols)
        rust_result = g.eval_to_float(rust_expr, 2.0, y.tolist(), [], [])
        pybamm_result = float(
            np.asarray(expr.evaluate(t=2.0, y=y.reshape(-1, 1))).flat[0]
        )
        assert rust_result == pytest.approx(pybamm_result, rel=1e-5)

    def test_conditional_branch_selection(self):
        """Test Conditional with different selector values selecting different branches."""
        import pybamm
        from pybamm.rust import ExprGraph

        selector = pybamm.InputParameter("s")
        branch1 = pybamm.Scalar(10.0)
        branch2 = pybamm.Scalar(20.0)
        branch3 = pybamm.Scalar(30.0)
        expr = pybamm.Conditional(selector, branch1, branch2, branch3)

        # Selector = 1.0 → branch 1 (10.0)
        g1 = ExprGraph()
        rust_symbols1 = {}
        re1 = expr.to_rust(g1, rust_symbols1)
        result1 = g1.eval_to_float(re1, 0.0, [], [], [1.0])
        assert result1 == pytest.approx(10.0)

        # Selector = 2.0 → branch 2 (20.0)
        g2 = ExprGraph()
        rust_symbols2 = {}
        re2 = expr.to_rust(g2, rust_symbols2)
        result2 = g2.eval_to_float(re2, 0.0, [], [], [2.0])
        assert result2 == pytest.approx(20.0)

        # Selector = 3.0 → branch 3 (30.0)
        g3 = ExprGraph()
        rust_symbols3 = {}
        re3 = expr.to_rust(g3, rust_symbols3)
        result3 = g3.eval_to_float(re3, 0.0, [], [], [3.0])
        assert result3 == pytest.approx(30.0)

        # Selector = 0.0 → no branch active, returns 0
        g4 = ExprGraph()
        rust_symbols4 = {}
        re4 = expr.to_rust(g4, rust_symbols4)
        result4 = g4.eval_to_float(re4, 0.0, [], [], [0.0])
        assert result4 == pytest.approx(0.0)

    def test_conditional_vs_python(self):
        """Test Conditional Rust evaluation matches Python evaluation."""
        import pybamm
        from pybamm.rust import ExprGraph

        selector = pybamm.InputParameter("s")
        branch1 = pybamm.InputParameter("a")
        branch2 = pybamm.InputParameter("b")
        expr = pybamm.Conditional(selector, branch1, branch2)

        for s_val in [0.5, 1.0, 1.5, 2.0, 2.5]:
            inputs = {"s": s_val, "a": 100.0, "b": 200.0}

            # Rust evaluation - inputs order matches expression traversal: s, a, b
            g = ExprGraph()
            rust_symbols = {}
            rust_expr = expr.to_rust(g, rust_symbols)
            inputs_list = [inputs["s"], inputs["a"], inputs["b"]]
            rust_result = g.eval_to_float(rust_expr, 0.0, [], [], inputs_list)

            # Python evaluation
            pybamm_result = float(expr.evaluate(inputs=inputs))

            assert rust_result == pytest.approx(pybamm_result, rel=1e-5)


class TestRustUnsupportedNodeErrors:
    """Unsupported nodes raise explicit, actionable errors naming CasADi."""

    def test_unsupported_4d_interpolant_message(self):
        import pybamm
        from pybamm.rust import ExprGraph

        # 2D/3D now lower natively; 4D+ keeps the actionable error.
        x = tuple(np.linspace(0, 1, 5) for _ in range(4))
        z = np.zeros((5, 5, 5, 5))
        svs = tuple(pybamm.StateVector(slice(i, i + 1)) for i in range(4))
        interp = pybamm.Interpolant(x, z, svs, interpolator="linear")
        with pytest.raises(NotImplementedError, match=r"convert_to_format"):
            interp.to_rust(ExprGraph(), {})

    def test_unsupported_base_symbol_message(self):
        import pybamm
        from pybamm.rust import ExprGraph

        # SpatialVariable has no _to_rust -> base Symbol._to_rust raises.
        sym = pybamm.SpatialVariable("x", domain=["negative electrode"])
        with pytest.raises(TypeError, match=r"convert_to_format") as exc:
            sym.to_rust(ExprGraph(), {})
        # Must point at the live API, not the removed IDAKLUSolver(evaluator=...).
        assert "evaluator=" not in str(exc.value)

    def test_unsupported_generic_function_message(self):
        import pybamm
        from pybamm.rust import ExprGraph

        # A generic Function wrapping a Python callable hits Function._rust_evaluate
        # rather than Symbol._to_rust, so it needs its own actionable message.
        sv = pybamm.StateVector(slice(0, 1))
        fun = pybamm.Function(np.sin, sv)
        with pytest.raises(TypeError, match=r"convert_to_format"):
            fun.to_rust(ExprGraph(), {})


class TestCrossGraphGuard:
    """An Expr from one ExprGraph must not be usable in another."""

    def test_builder_rejects_foreign_expr(self):
        import pytest

        from pybamm.rust import ExprGraph

        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        b = g_b.scalar(3.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            g_a.add(a, b)  # b belongs to g_b

    def test_dunder_rejects_foreign_expr(self):
        import pytest

        from pybamm.rust import ExprGraph

        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        b = g_b.scalar(3.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            _ = a + b  # __add__ on a, other from g_b

    def test_compile_rejects_foreign_expr(self):
        import pytest

        from pybamm.rust import ExprGraph

        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            g_b.compile(a)  # a belongs to g_a

    def test_compile_group_rejects_foreign_expr(self):
        import pytest

        from pybamm.rust import ExprGraph

        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            g_b.compile_group({"x": a})  # a belongs to g_a


class TestRustReduceSubgradientParity:
    """pybamm.max/min Rust Jacobian matches casadi.mmax/mmin (argmax indicator)."""

    def _rust_jac_row(self, expr_pybamm, n, y):
        import numpy as np

        from pybamm.rust import ExprGraph

        g = ExprGraph()
        expr = expr_pybamm.to_rust(g, {})
        f = g.compile(expr, n_states=n)
        return f.jacobian("y")(0.0, y, np.array([])).toarray()

    def test_max_jacobian_matches_casadi(self):
        import casadi
        import numpy as np

        import pybamm

        n = 4
        y = np.array([0.3, 0.9, 0.1, 0.5])  # unique argmax at index 1
        sv = pybamm.StateVector(slice(0, n))
        rust_row = self._rust_jac_row(pybamm.max(sv), n, y)

        ys = casadi.MX.sym("y", n)
        jac = casadi.Function("J", [ys], [casadi.jacobian(casadi.mmax(ys), ys)])
        casadi_row = np.array(jac(y)).reshape(1, n)

        expected = np.zeros((1, n))
        expected[0, 1] = 1.0
        np.testing.assert_allclose(rust_row, casadi_row, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(rust_row, expected, rtol=1e-12, atol=1e-12)

    def test_min_jacobian_matches_casadi(self):
        import casadi
        import numpy as np

        import pybamm

        n = 4
        y = np.array([0.3, 0.9, 0.1, 0.5])  # unique argmin at index 2
        sv = pybamm.StateVector(slice(0, n))
        rust_row = self._rust_jac_row(pybamm.min(sv), n, y)

        ys = casadi.MX.sym("y", n)
        jac = casadi.Function("J", [ys], [casadi.jacobian(casadi.mmin(ys), ys)])
        casadi_row = np.array(jac(y)).reshape(1, n)

        expected = np.zeros((1, n))
        expected[0, 2] = 1.0
        np.testing.assert_allclose(rust_row, casadi_row, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(rust_row, expected, rtol=1e-12, atol=1e-12)
