import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csc_matrix, csr_matrix


class TestRustExprGraph:
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
    def _eval_scalar(self, pybamm_expr, g=None):
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

        p = pybamm.InputParameter("x")
        expr = pybamm.erf(p)
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr.to_rust(g, rust_symbols)
        result = g.eval_to_float(rust_expr, 0.0, [], [], [1.0])
        expected = float(expr.evaluate(inputs={"x": 1.0}))
        assert result == pytest.approx(expected)

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
            (0.0, 0.5),
            # Inner exponents (a - 1) / 2 of 0, -1, 0.25, 0.5 and 1, and a runtime pow.
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

    @pytest.mark.parametrize("exponent", [0.5, 2.0])
    @pytest.mark.parametrize("scale", [1.0, 33133.0])
    def test_reg_power_is_bitwise_identical_across_backends(self, exponent, scale):
        # These exponents use only operations IEEE 754 rounds correctly, so NumPy,
        # CasADi and Rust must agree bit for bit.
        import casadi

        import pybamm
        from pybamm.rust import ExprGraph

        magnitudes = np.concatenate([[0.0, 5e-324], np.geomspace(1e-12, 2.0, 2000)])
        base = scale * np.concatenate([magnitudes, -magnitudes])
        y = pybamm.StateVector(slice(0, base.size))
        expr = pybamm.reg_power(y, exponent, scale=scale)

        reference = expr.evaluate(y=base).ravel()
        y_casadi = casadi.MX.sym("y", base.size)
        f = casadi.Function("f", [y_casadi], [expr.to_casadi(y=y_casadi)])
        g = ExprGraph()
        compiled = g.compile(expr.to_rust(g, {}), name="f", n_states=base.size)
        backends = {
            "CasADi MX": f(base),
            "CasADi SX": f.expand()(base),
            "Rust": compiled(0.0, base, np.array([])),
        }
        for name, values in backends.items():
            np.testing.assert_array_equal(
                np.asarray(values, dtype=np.float64).ravel().view(np.uint64),
                reference.view(np.uint64),
                err_msg=name,
            )

    @pytest.mark.parametrize("exponent", [0.5, 2.0])
    def test_reg_power_with_zero_delta_matches_casadi(self, exponent):
        import casadi

        import pybamm
        from pybamm.rust import ExprGraph

        base = np.array([0.0, -4.0, 4.0])
        y = pybamm.StateVector(slice(0, base.size))
        expr = pybamm.RegPower(y, exponent, delta=0)
        y_casadi = casadi.MX.sym("y", base.size)
        casadi_expr = expr.to_casadi(y=y_casadi)
        f = casadi.Function("f", [y_casadi], [casadi_expr])
        jac = casadi.Function("J", [y_casadi], [casadi.jacobian(casadi_expr, y_casadi)])
        g = ExprGraph()
        compiled = g.compile(expr.to_rust(g), n_states=base.size)

        np.testing.assert_array_equal(
            compiled(0.0, base, np.array([])), np.asarray(f(base)).ravel()
        )
        np.testing.assert_array_equal(
            np.diag(compiled.jacobian("y")(0.0, base, np.array([])).toarray()),
            np.diag(np.asarray(jac(base))),
        )


class TestIndexToRust:
    def test_index_vector_slice(self):
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
        import pybamm
        from pybamm.rust import ExprGraph

        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr_pybamm = pybamm.Index(v, -1)
        g = ExprGraph()
        rust_expr = expr_pybamm.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        expected = expr_pybamm.evaluate().flatten()
        assert_array_almost_equal(result, expected)

    def test_index_step_raises(self):
        import pybamm
        from pybamm.rust import ExprGraph

        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr_pybamm = pybamm.Index(v, slice(0, 4, 2))
        g = ExprGraph()
        with pytest.raises(NotImplementedError, match="step"):
            expr_pybamm.to_rust(g, {})

    def test_index_over_negative_start_clamps(self):
        import pybamm
        from pybamm.rust import ExprGraph

        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr_pybamm = pybamm.Index(v, slice(-7, 5))
        g = ExprGraph()
        rust_expr = expr_pybamm.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        expected = expr_pybamm.evaluate().flatten()
        assert_array_almost_equal(result, expected)


class TestNewBinaryOperatorsToRust:
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

    @pytest.mark.parametrize(
        ("a", "b"), [(-1.0, 3.0), (1.0, -3.0), (-7.5, 2.0), (-7.0, -3.0)]
    )
    def test_modulo_with_negative_operands_truncates_like_casadi(self, a, b):
        # evaluate() takes the floored remainder; Rust keeps CasADi's fmod
        import casadi

        import pybamm

        expr = pybamm.Modulo(pybamm.InputParameter("a"), pybamm.InputParameter("b"))
        inputs = {"a": a, "b": b}
        result = self._eval_scalar(expr, inputs=inputs)
        expected = float(casadi.evalf(expr.to_casadi(inputs=inputs)))
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
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.EqualHeaviside(a, b)
        result = self._eval_scalar(expr, inputs={"a": 3.0, "b": 5.0})
        assert result == pytest.approx(1.0)

    def test_equal_heaviside_equal(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.EqualHeaviside(a, b)
        result = self._eval_scalar(expr, inputs={"a": 5.0, "b": 5.0})
        assert result == pytest.approx(1.0)

    def test_equal_heaviside_false(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.EqualHeaviside(a, b)
        result = self._eval_scalar(expr, inputs={"a": 7.0, "b": 5.0})
        assert result == pytest.approx(0.0)

    def test_not_equal_heaviside_true(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        expr = pybamm.NotEqualHeaviside(a, b)
        result = self._eval_scalar(expr, inputs={"a": 3.0, "b": 5.0})
        assert result == pytest.approx(1.0)

    def test_not_equal_heaviside_equal(self):
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
        from scipy.sparse import csr_matrix as scipy_csr

        import pybamm
        from pybamm.rust import ExprGraph

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
    def test_1d_linear(self):
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
        import pybamm
        from pybamm.rust import ExprGraph

        x = np.linspace(0.0, 1.0, 11)
        y = 3.0 * x + 1.0
        sv = pybamm.StateVector(slice(0, 2))
        interp = pybamm.Interpolant(x, y, sv)

        y_test = np.array([-0.5, 1.5])
        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, y_test, np.array([]), [])
        expected = interp.evaluate(y=y_test.reshape(-1, 1)).flatten()
        assert_array_almost_equal(result, expected, decimal=10)

    def test_1d_linear_repeated_knot_matches_casadi(self):
        import casadi

        import pybamm
        from pybamm.rust import ExprGraph

        # A step at t = 600, which CasADi takes as right-continuous
        x = np.array([0.0, 600.0, 600.0, 1200.0])
        interp = pybamm.Interpolant(x, np.array([1.0, 1.0, 2.0, 2.0]), pybamm.t)

        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        t_sym = casadi.MX.sym("t")
        casadi_interp = casadi.Function("f", [t_sym], [interp.to_casadi(t=t_sym)])
        times = [-10.0, 0.0, 599.9, 600.0, 600.1, 1200.0, 1300.0]
        result = [g.eval_to_float(rust_expr, t, [], [], []) for t in times]
        expected = [float(casadi_interp(t)) for t in times]
        np.testing.assert_array_equal(result, expected)
        assert result[3] == pytest.approx(2.0)

    def test_1d_linear_refused_table_raises_model_error(self):
        import pybamm
        from pybamm.rust import ExprGraph

        # The first segment extrapolates, so it may not have zero width
        x = np.array([0.0, 0.0, 1.0])
        interp = pybamm.Interpolant(x, np.array([1.0, 2.0, 3.0]), pybamm.t)
        with pytest.raises(pybamm.ModelError, match=r"repeat.*convert_to_format"):
            interp.to_rust(ExprGraph(), {})

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
        import pybamm
        from pybamm.rust import ExprGraph

        x = np.linspace(0.0, 4.0, 20)
        y = np.sin(x)
        sv = pybamm.StateVector(slice(0, 3))
        interp = pybamm.Interpolant(x, y, sv, interpolator="cubic")

        y_test = np.array([0.7, 2.1, 5.0])
        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, y_test, np.array([]), [])
        expected = interp.evaluate(y=y_test.reshape(-1, 1)).flatten()
        assert_array_almost_equal(result, expected, decimal=10)

    @pytest.mark.parametrize("interpolator", ["linear", "cubic", "pchip"])
    def test_1d_extrapolate_false_extends(self, interpolator):
        """Rust ignores extrapolate=False: a NaN would poison the solver residual."""
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
        yq = np.array([1.3])
        result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
        expected = interp.evaluate(y=yq.reshape(-1, 1)).flatten()
        assert_array_almost_equal(result, expected, decimal=10)
        for q in [-1.0, 7.5]:
            yq = np.array([q])
            assert np.isnan(interp.evaluate(y=yq.reshape(-1, 1))).all()
            result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
            expected = extend.evaluate(y=yq.reshape(-1, 1)).flatten()
            assert_array_almost_equal(result, expected, decimal=10)

    @pytest.mark.parametrize("interpolator", ["linear", "cubic", "pchip"])
    def test_1d_vector_valued_y_matches_evaluate(self, interpolator):
        import pybamm
        from pybamm.rust import ExprGraph

        # Distinct nonlinear columns so a swapped or short table cannot cancel out.
        x = np.array([0.0, 0.5, 1.0, 2.0, 3.5, 5.0])
        y = np.column_stack([np.sin(x), np.cos(x), x**2])
        sv = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(x, y, sv, interpolator=interpolator)

        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        for q in [-1.0, 0.0, 0.5, 1.3, 2.0, 3.5, 5.0, 7.5]:
            yq = np.array([q])
            result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
            expected = interp.evaluate(y=yq.reshape(-1, 1)).flatten()
            assert_array_almost_equal(result, expected, decimal=10)

    @pytest.mark.parametrize("interpolator", ["linear", "cubic", "pchip"])
    def test_1d_vector_valued_y_jacobian_matches_casadi(self, interpolator):
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
        import pybamm
        from pybamm.rust import ExprGraph

        x = np.linspace(0.0, 1.0, 5)
        y = np.ones((5, 2, 2))
        sv = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(x, y, sv)
        with pytest.raises(NotImplementedError, match=r"more than two dimensions"):
            interp.to_rust(ExprGraph(), {})


class TestInterpolantNDToRust:
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
        outs.append([a[0] - 0.8 for a in axes])
        outs.append([a[-1] + 1.1 for a in axes])
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
        import pybamm
        from pybamm.rust import ExprGraph

        X0, X1 = np.meshgrid(self.x0, self.x1, indexing="ij")
        y = X0**2 + X1
        sv0 = pybamm.StateVector(slice(0, 3))
        sv1 = pybamm.StateVector(slice(3, 6))
        interp = pybamm.Interpolant(
            (self.x0, self.x1), y, (sv0, sv1), interpolator=interpolator
        )
        y_test = np.array([0.7, 2.1, 5.0, 0.0, 1.5, -1.4])
        g = ExprGraph()
        rust_expr = interp.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, y_test, np.array([]), [])
        expected = np.asarray(interp.evaluate(y=y_test.reshape(-1, 1))).flatten()
        assert_array_almost_equal(result, expected, decimal=10)

    @pytest.mark.parametrize("interpolator", ["linear", "cubic"])
    def test_2d_extrapolate_false_extends(self, interpolator):
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
        yq = np.array([1.3, 0.5])
        result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
        expected = np.asarray(interp.evaluate(y=yq.reshape(-1, 1))).flatten()
        assert_array_almost_equal(result, expected, decimal=10)
        for p in [[-0.7, 0.5], [4.9, 3.1]]:
            yq = np.asarray(p)
            assert np.isnan(interp.evaluate(y=yq.reshape(-1, 1))).all()
            result = g.eval_to_array(rust_expr, 0.0, yq, np.array([]), [])
            expected = np.asarray(extend.evaluate(y=yq.reshape(-1, 1))).flatten()
            assert_array_almost_equal(result, expected, decimal=10)


class TestReductionToRust:
    def test_max_reduce(self):
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
    def test_numpy_concatenation(self):
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

    def test_domain_concatenation_repeated_over_a_secondary_domain(self):
        import pybamm
        from pybamm.rust import ExprGraph
        from tests import get_1p1d_discretisation_for_testing

        disc = get_1p1d_discretisation_for_testing(zpts=4)
        secondary = {"secondary": "current collector"}
        negative_and_separator = pybamm.Variable(
            "a", domain=["negative electrode", "separator"], auxiliary_domains=secondary
        )
        positive = pybamm.Variable(
            "b", domain=["positive electrode"], auxiliary_domains=secondary
        )
        disc.set_variable_slices([negative_and_separator, positive])
        expr = disc.process_symbol(
            pybamm.concatenation(2 * negative_and_separator, positive**2)
        )
        assert isinstance(expr, pybamm.DomainConcatenation)
        assert expr.secondary_dimensions_npts == 4

        y = np.linspace(-1.0, 1.0, expr.size)
        g = ExprGraph()
        result = g.eval_to_array(expr.to_rust(g), 0.0, y, np.array([]), [])
        np.testing.assert_array_equal(result, expr.evaluate(y=y).ravel())


class TestVectorFieldToRust:
    def test_vector_field(self):
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
    def _assert_rust_eval_matches(self, expr, t=0.0, y=None, y_dot=None, inputs=None):
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr.to_rust(g, rust_symbols)

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

        from pybamm.rust import ExprGraph

        expr_erf = pybamm.erf(p)
        g = ExprGraph()
        rust_symbols = {}
        re = expr_erf.to_rust(g, rust_symbols)
        rust_result = g.eval_to_float(re, 0.0, [], [], [1.0])
        pybamm_result = float(expr_erf.evaluate(inputs={"x": 1.0}))
        assert rust_result == pytest.approx(pybamm_result)

    def test_floor_ceiling(self):
        import pybamm

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
        import pybamm

        p = pybamm.InputParameter("x")
        expr = pybamm.sign(p)

        from pybamm.rust import ExprGraph

        g = ExprGraph()
        rust_symbols = {}
        re = expr.to_rust(g, rust_symbols)
        rust_result = g.eval_to_float(re, 0.0, [], [], [5.0])
        pybamm_result = float(expr.evaluate(inputs={"x": 5.0}))
        assert rust_result == pytest.approx(pybamm_result)

        g2 = ExprGraph()
        rust_symbols2 = {}
        re2 = expr.to_rust(g2, rust_symbols2)
        rust_result2 = g2.eval_to_float(re2, 0.0, [], [], [-3.0])
        pybamm_result2 = float(expr.evaluate(inputs={"x": -3.0}))
        assert rust_result2 == pytest.approx(pybamm_result2)

        g3 = ExprGraph()
        rust_symbols3 = {}
        re3 = expr.to_rust(g3, rust_symbols3)
        rust_zero = g3.eval_to_float(re3, 0.0, [], [], [0.0])
        pybamm_zero = float(expr.evaluate(inputs={"x": 0.0}))
        assert rust_zero == pytest.approx(pybamm_zero)

    def test_modulo_minimum_maximum(self):
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
        import pybamm

        sv = pybamm.StateVector(slice(0, 3))
        expr = pybamm.Scalar(2.0) * sv + pybamm.Scalar(1.0)
        y = np.array([1.0, 2.0, 3.0])
        self._assert_rust_eval_matches(expr, y=y.reshape(-1, 1))

    def test_multi_slice_state_vector_and_dot(self):
        import pybamm

        sv = pybamm.StateVector(slice(0, 2), slice(4, 6))
        y = np.arange(6.0)
        self._assert_rust_eval_matches(sv, y=y.reshape(-1, 1))

        sv_dot = pybamm.StateVectorDot(slice(0, 2), slice(4, 6))
        y_dot = np.arange(10.0, 16.0)
        self._assert_rust_eval_matches(sv_dot, y_dot=y_dot.reshape(-1, 1))

    def test_heaviside(self):
        import pybamm

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")

        from pybamm.rust import ExprGraph

        expr_eh = pybamm.EqualHeaviside(a, b)
        g = ExprGraph()
        rust_symbols = {}
        re = expr_eh.to_rust(g, rust_symbols)
        rust_result = g.eval_to_float(re, 0.0, [], [], [2.0, 3.0])
        pybamm_result = float(expr_eh.evaluate(inputs={"a": 2.0, "b": 3.0}))
        assert rust_result == pytest.approx(pybamm_result)
        assert rust_result == pytest.approx(1.0)

        expr_neh = pybamm.NotEqualHeaviside(a, b)
        g2 = ExprGraph()
        rust_symbols2 = {}
        re2 = expr_neh.to_rust(g2, rust_symbols2)
        rust_result2 = g2.eval_to_float(re2, 0.0, [], [], [3.0, 3.0])
        pybamm_result2 = float(expr_neh.evaluate(inputs={"a": 3.0, "b": 3.0}))
        assert rust_result2 == pytest.approx(pybamm_result2)
        assert rust_result2 == pytest.approx(0.0)

    def test_interpolation_1d_linear(self):
        import pybamm

        x = np.linspace(0, 1, 50)
        y_data = 2 * x
        sv = pybamm.StateVector(slice(0, 2))
        interp = pybamm.Interpolant(x, y_data, sv)

        y_test = np.array([0.4, 0.6])
        self._assert_rust_eval_matches(interp, y=y_test.reshape(-1, 1))

    def test_max_min_reduction(self):
        import pybamm

        v = pybamm.Vector(np.array([1.0, 5.0, 3.0]))
        self._assert_rust_eval_matches(pybamm.max(v))
        self._assert_rust_eval_matches(pybamm.min(v))

    def test_index_slice(self):
        import pybamm

        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr = pybamm.Index(v, slice(1, 3))
        self._assert_rust_eval_matches(expr)

    def test_concatenation(self):
        import pybamm

        v1 = pybamm.Vector(np.array([1.0, 2.0]))
        v2 = pybamm.Vector(np.array([3.0, 4.0, 5.0]))
        expr = pybamm.NumpyConcatenation(v1, v2)
        self._assert_rust_eval_matches(expr)

    def test_sparse_matmul(self):
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

    @pytest.mark.parametrize(
        "entries",
        [
            csc_matrix((1, 1)),
            csr_matrix(([2.0], ([1], [0])), shape=(3, 1)),
            csr_matrix(([2.0, -1.0], ([0, 1], [1, 0])), shape=(2, 2)),
        ],
        ids=["all-zero", "column", "square"],
    )
    def test_sparse_matrix_as_value(self, entries):
        import pybamm
        from pybamm.rust import ExprGraph

        g = ExprGraph()
        expr = pybamm.Matrix(entries).to_rust(g)
        result = g.eval_to_array(expr, 0.0, np.array([]), np.array([]), [])
        np.testing.assert_array_equal(result, entries.toarray().ravel())

    def test_sparse_matrix_in_elementwise_product(self):
        import pybamm
        from pybamm.rust import ExprGraph

        mat = pybamm.Matrix(csr_matrix(([2.0], ([1], [0])), shape=(3, 1)))
        expr = mat * pybamm.StateVector(slice(0, 1))
        g = ExprGraph()
        result = g.eval_to_array(
            expr.to_rust(g), 0.0, np.array([3.0]), np.array([]), []
        )
        np.testing.assert_array_equal(result, [0.0, 6.0, 0.0])

    def test_sparse_matrix_as_matmul_operand_and_value(self):
        import pybamm
        from pybamm.rust import ExprGraph

        mat = pybamm.Matrix(csr_matrix(([2.0], ([1], [0])), shape=(3, 1)))
        sv = pybamm.StateVector(slice(0, 1))
        g = ExprGraph()
        rust_symbols = {}
        value = mat.to_rust(g, rust_symbols)
        product = (mat @ sv).to_rust(g, rust_symbols)
        y = np.array([3.0])
        np.testing.assert_array_equal(
            g.eval_to_array(value, 0.0, y, np.array([]), []), [0.0, 2.0, 0.0]
        )
        np.testing.assert_array_equal(
            g.eval_to_array(product, 0.0, y, np.array([]), []), [0.0, 6.0, 0.0]
        )

    def test_sparse_matmul_sums_rows_in_column_order(self):
        from fractions import Fraction

        import pybamm
        from pybamm.rust import ExprGraph

        # A stencil row stored out of column order; summing the two outer
        # entries first rounds differently, and breaks conservation
        a = 33.85469387755101
        entries = csr_matrix(
            (np.array([a, a, -2 * a]), np.array([2, 0, 1]), np.array([0, 3])),
            shape=(1, 3),
        )
        y = np.array([4.05, 4.0496, 4.0491])
        expr = pybamm.Matrix(entries) @ pybamm.StateVector(slice(0, 3))
        g = ExprGraph()
        result = g.eval_to_array(expr.to_rust(g), 0.0, y, np.array([]), [])

        def fused(x, factor, acc):
            # Correctly rounded, as one FMA instruction computes it
            return float(Fraction(x) * Fraction(factor) + Fraction(acc))

        # Fused where the CPU has an FMA instruction; either way, in column order
        column_order = [
            fused(a, y[2], fused(-2 * a, y[1], a * y[0])),
            (a * y[0] + (-2 * a) * y[1]) + a * y[2],
        ]
        assert result[0] in column_order
        # The matrix the symbol holds keeps its own order
        np.testing.assert_array_equal(entries.indices, [2, 0, 1])

    def test_composed_expression(self):
        import pybamm

        sv = pybamm.StateVector(slice(0, 1))
        t_sym = pybamm.t
        expr = pybamm.sqrt(sv**2 + pybamm.Scalar(1.0)) * pybamm.exp(-t_sym)

        y = np.array([3.0])
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
        import pybamm
        from pybamm.rust import ExprGraph

        selector = pybamm.InputParameter("s")
        branch1 = pybamm.Scalar(10.0)
        branch2 = pybamm.Scalar(20.0)
        branch3 = pybamm.Scalar(30.0)
        expr = pybamm.Conditional(selector, branch1, branch2, branch3)

        g1 = ExprGraph()
        rust_symbols1 = {}
        re1 = expr.to_rust(g1, rust_symbols1)
        result1 = g1.eval_to_float(re1, 0.0, [], [], [1.0])
        assert result1 == pytest.approx(10.0)

        g2 = ExprGraph()
        rust_symbols2 = {}
        re2 = expr.to_rust(g2, rust_symbols2)
        result2 = g2.eval_to_float(re2, 0.0, [], [], [2.0])
        assert result2 == pytest.approx(20.0)

        g3 = ExprGraph()
        rust_symbols3 = {}
        re3 = expr.to_rust(g3, rust_symbols3)
        result3 = g3.eval_to_float(re3, 0.0, [], [], [3.0])
        assert result3 == pytest.approx(30.0)

        g4 = ExprGraph()
        rust_symbols4 = {}
        re4 = expr.to_rust(g4, rust_symbols4)
        result4 = g4.eval_to_float(re4, 0.0, [], [], [0.0])
        assert result4 == pytest.approx(0.0)

    def test_conditional_vs_python(self):
        import pybamm
        from pybamm.rust import ExprGraph

        selector = pybamm.InputParameter("s")
        branch1 = pybamm.InputParameter("a")
        branch2 = pybamm.InputParameter("b")
        expr = pybamm.Conditional(selector, branch1, branch2)

        for s_val in [0.5, 1.0, 1.5, 2.0, 2.5]:
            inputs = {"s": s_val, "a": 100.0, "b": 200.0}

            # Inputs are ordered by expression traversal: s, a, b
            g = ExprGraph()
            rust_symbols = {}
            rust_expr = expr.to_rust(g, rust_symbols)
            inputs_list = [inputs["s"], inputs["a"], inputs["b"]]
            rust_result = g.eval_to_float(rust_expr, 0.0, [], [], inputs_list)

            pybamm_result = float(expr.evaluate(inputs=inputs))

            assert rust_result == pytest.approx(pybamm_result, rel=1e-5)


class TestRustUnsupportedNodeErrors:
    def test_unsupported_4d_interpolant_message(self):
        import pybamm
        from pybamm.rust import ExprGraph

        x = tuple(np.linspace(0, 1, 5) for _ in range(4))
        z = np.zeros((5, 5, 5, 5))
        svs = tuple(pybamm.StateVector(slice(i, i + 1)) for i in range(4))
        interp = pybamm.Interpolant(x, z, svs, interpolator="linear")
        with pytest.raises(NotImplementedError, match=r"convert_to_format"):
            interp.to_rust(ExprGraph(), {})

    def test_unsupported_base_symbol_message(self):
        import pybamm
        from pybamm.rust import ExprGraph

        sym = pybamm.SpatialVariable("x", domain=["negative electrode"])
        with pytest.raises(TypeError, match=r"convert_to_format") as exc:
            sym.to_rust(ExprGraph(), {})
        # The fix it suggests is a model setting, not a solver argument
        assert "evaluator=" not in str(exc.value)

    def test_unsupported_generic_function_message(self):
        import pybamm
        from pybamm.rust import ExprGraph

        sv = pybamm.StateVector(slice(0, 1))
        fun = pybamm.Function(np.sin, sv)
        with pytest.raises(TypeError, match=r"convert_to_format"):
            fun.to_rust(ExprGraph(), {})

    def test_specific_function_without_a_rust_form_message(self):
        import pybamm
        from pybamm.rust import ExprGraph

        fun = pybamm.SpecificFunction(np.abs, pybamm.StateVector(slice(0, 1)))
        with pytest.raises(TypeError, match=r"no native Rust implementation"):
            fun.to_rust(ExprGraph(), {})

    @pytest.mark.parametrize(
        "name",
        [
            "KroneckerProduct",
            "TensorProduct",
            "Transpose",
            "Magnitude",
            "Component",
            "Norm",
            "Gradient",
        ],
    )
    def test_unsupported_operator_message(self, name):
        import pybamm
        from pybamm.rust import ExprGraph

        y = pybamm.StateVector(slice(0, 2))
        field = pybamm.VectorField(y, pybamm.StateVector(slice(2, 4)))
        identity = pybamm.Matrix(np.eye(2))
        nodes = {
            "KroneckerProduct": pybamm.KroneckerProduct(identity, identity),
            "TensorProduct": pybamm.TensorProduct(y, y),
            "Transpose": pybamm.Transpose(y),
            "Magnitude": pybamm.Magnitude(field, "lr"),
            "Component": pybamm.Component(field, 0),
            "Norm": pybamm.Norm(field),
            "Gradient": pybamm.Gradient(
                pybamm.StateVector(slice(0, 3), domain="negative electrode")
            ),
        }
        with pytest.raises(TypeError, match=rf"'{name}'.*convert_to_format"):
            nodes[name].to_rust(ExprGraph(), {})


class TestCrossGraphGuard:
    def test_builder_rejects_foreign_expr(self):
        import pytest

        from pybamm.rust import ExprGraph

        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        b = g_b.scalar(3.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            g_a.add(a, b)

    def test_dunder_rejects_foreign_expr(self):
        import pytest

        from pybamm.rust import ExprGraph

        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        b = g_b.scalar(3.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            _ = a + b

    def test_compile_rejects_foreign_expr(self):
        import pytest

        from pybamm.rust import ExprGraph

        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            g_b.compile(a)

    def test_compile_outputs_rejects_foreign_expr(self):
        import pytest

        from pybamm.rust import ExprGraph

        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            g_b.compile_outputs({"x": a})


class TestRustSubgradientParity:
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

    @pytest.mark.parametrize("operator", ["abs", "sign"])
    def test_derivative_at_zero_matches_casadi(self, operator):
        import casadi

        import pybamm

        y = np.array([0.0])
        sv = pybamm.StateVector(slice(0, 1))
        expr = abs(sv) if operator == "abs" else pybamm.sign(sv)
        rust_row = self._rust_jac_row(expr, 1, y)

        ys = casadi.MX.sym("y", 1)
        jac = casadi.Function("J", [ys], [casadi.jacobian(expr.to_casadi(y=ys), ys)])
        np.testing.assert_array_equal(rust_row, np.asarray(jac(y)).reshape(1, 1))
        np.testing.assert_array_equal(rust_row, [[0.0]])

    @pytest.mark.parametrize("operator", ["minimum", "maximum"])
    def test_tied_binary_operands_take_the_left_derivative(self, operator):
        # As Symbol.jac does; CasADi's fmin and fmax split a tie evenly instead
        import pybamm

        y = np.array([0.5, 0.5])
        left = pybamm.StateVector(slice(0, 1))
        right = pybamm.StateVector(slice(1, 2))
        expr = getattr(pybamm, operator)(left, right)
        rust_row = self._rust_jac_row(expr, 2, y)

        symbolic_row = expr.jac(pybamm.StateVector(slice(0, 2))).evaluate(y=y)
        np.testing.assert_array_equal(rust_row, symbolic_row.toarray())
        np.testing.assert_array_equal(rust_row, [[1.0, 0.0]])

    @pytest.mark.parametrize("reduction", ["max", "min"])
    def test_tied_reduction_takes_the_first_occurrence(self, reduction):
        # CasADi's mmax and mmin split a tie evenly instead
        import pybamm

        y = np.array([0.3, 0.5, 0.5, 0.3])
        if reduction == "min":
            y = -y
        sv = pybamm.StateVector(slice(0, 4))
        rust_row = self._rust_jac_row(getattr(pybamm, reduction)(sv), 4, y)
        np.testing.assert_array_equal(rust_row, [[0.0, 1.0, 0.0, 0.0]])
