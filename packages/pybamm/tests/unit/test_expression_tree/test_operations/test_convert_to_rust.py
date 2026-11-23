import itertools
from fractions import Fraction

import casadi
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csc_matrix, csr_matrix

import pybamm
from pybamm.rust import ExprGraph
from tests import get_1p1d_discretisation_for_testing

pytest_plugins = ["tests.conftest_rust"]


class TestRustExprGraph:
    def test_import(self):
        from pybamm.rust import Expr  # noqa: F401

    def test_scalar(self):
        g = ExprGraph()
        s = g.scalar(3.14)
        result = g.eval_to_float(s, 0.0, [], [], [])
        assert result == pytest.approx(3.14)

    def test_time(self):
        g = ExprGraph()
        t = g.time()
        result = g.eval_to_float(t, 2.5, [], [], [])
        assert result == pytest.approx(2.5)

    def test_state_vector(self):
        g = ExprGraph()
        sv = g.state_vector(1, 3)
        y = np.array([10.0, 20.0, 30.0, 40.0])
        result = g.eval_to_array(sv, 0.0, y, np.array([]), [])
        assert_array_almost_equal(result, [20.0, 30.0])

    def test_state_vector_dot(self):
        g = ExprGraph()
        sv_dot = g.state_vector_dot(0, 2)
        y_dot = np.array([100.0, 200.0, 300.0])
        result = g.eval_to_array(sv_dot, 0.0, np.array([]), y_dot, [])
        assert_array_almost_equal(result, [100.0, 200.0])

    def test_input_parameter(self):
        g = ExprGraph()
        p = g.input_parameter("C_rate")
        result = g.eval_to_float(p, 0.0, [], [], [1.5])
        assert result == pytest.approx(1.5)

    def test_array(self):
        g = ExprGraph()
        a = g.array(np.array([1.0, 2.0, 3.0]))
        result = g.eval_to_array(a, 0.0, np.array([]), np.array([]), [])
        assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_add_scalars(self):
        g = ExprGraph()
        a = g.scalar(2.0)
        b = g.scalar(3.0)
        c = g.add(a, b)
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(5.0)

    def test_add_dunder(self):
        g = ExprGraph()
        a = g.scalar(2.0)
        b = g.scalar(3.0)
        c = a + b
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(5.0)

    def test_mul_dunder(self):
        g = ExprGraph()
        a = g.scalar(4.0)
        b = g.scalar(5.0)
        c = a * b
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(20.0)

    def test_sub_dunder(self):
        g = ExprGraph()
        a = g.scalar(10.0)
        b = g.scalar(3.0)
        c = a - b
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(7.0)

    def test_neg_dunder(self):
        g = ExprGraph()
        a = g.scalar(5.0)
        b = -a
        result = g.eval_to_float(b, 0.0, [], [], [])
        assert result == pytest.approx(-5.0)

    def test_truediv_dunder(self):
        g = ExprGraph()
        a = g.scalar(10.0)
        b = g.scalar(4.0)
        c = a / b
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(2.5)

    def test_pow_dunder(self):
        g = ExprGraph()
        a = g.scalar(3.0)
        b = g.scalar(2.0)
        c = a**b
        result = g.eval_to_float(c, 0.0, [], [], [])
        assert result == pytest.approx(9.0)

    def test_nested_expression(self):
        g = ExprGraph()
        two = g.scalar(2.0)
        three = g.scalar(3.0)
        four = g.scalar(4.0)
        one = g.scalar(1.0)
        result_expr = (two + three) * four - one
        result = g.eval_to_float(result_expr, 0.0, [], [], [])
        assert result == pytest.approx(19.0)

    def test_vector_add(self):
        g = ExprGraph()
        a = g.array(np.array([1.0, 2.0, 3.0]))
        b = g.array(np.array([10.0, 20.0, 30.0]))
        c = a + b
        result = g.eval_to_array(c, 0.0, np.array([]), np.array([]), [])
        assert_array_almost_equal(result, [11.0, 22.0, 33.0])

    def test_scalar_times_vector(self):
        g = ExprGraph()
        s = g.scalar(2.0)
        v = g.array(np.array([1.0, 2.0, 3.0]))
        c = s * v
        result = g.eval_to_array(c, 0.0, np.array([]), np.array([]), [])
        assert_array_almost_equal(result, [2.0, 4.0, 6.0])


class TestSymbolToRust:
    def test_scalar_to_rust(self):
        s = pybamm.Scalar(3.14)
        g = ExprGraph()
        rust_symbols = {}
        expr = s.to_rust(g, rust_symbols)
        result = g.eval_to_float(expr, 0.0, [], [], [])
        assert result == pytest.approx(s.evaluate())

    def test_array_to_rust(self):
        a = pybamm.Vector(np.array([1.0, 2.0, 3.0]))
        g = ExprGraph()
        rust_symbols = {}
        expr = a.to_rust(g, rust_symbols)
        result = g.eval_to_array(expr, 0.0, np.array([]), np.array([]), [])
        expected = a.evaluate()
        assert_array_almost_equal(result, expected.flatten())

    def test_state_vector_to_rust(self):
        sv = pybamm.StateVector(slice(1, 4))
        y = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        g = ExprGraph()
        rust_symbols = {}
        expr = sv.to_rust(g, rust_symbols)
        result = g.eval_to_array(expr, 0.0, y, np.array([]), [])
        expected = sv.evaluate(y=y)
        assert_array_almost_equal(result, expected.flatten())

    def test_caching(self):
        s = pybamm.Scalar(3.14)
        g = ExprGraph()
        rust_symbols = {}
        expr1 = s.to_rust(g, rust_symbols)
        expr2 = s.to_rust(g, rust_symbols)
        assert expr1.id == expr2.id


class TestRegPowerToRust:
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
        p = pybamm.InputParameter("x")
        expr = pybamm.reg_power(p, exponent)
        g = ExprGraph()
        rust_expr = expr.to_rust(g, {})
        result = g.eval_to_float(rust_expr, 0.0, [], [], [x_val])
        expected = float(expr.evaluate(inputs={"x": x_val}))
        assert result == pytest.approx(expected, rel=1e-12, abs=1e-15)

    def test_reg_power_with_scale(self):
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
        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr_pybamm = pybamm.Index(v, slice(1, 3))
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        expected = expr_pybamm.evaluate().flatten()
        assert_array_almost_equal(result, expected)

    def test_index_negative_last_element(self):
        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr_pybamm = pybamm.Index(v, -1)
        g = ExprGraph()
        rust_expr = expr_pybamm.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        expected = expr_pybamm.evaluate().flatten()
        assert_array_almost_equal(result, expected)

    def test_index_step_raises(self):
        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr_pybamm = pybamm.Index(v, slice(0, 4, 2))
        g = ExprGraph()
        with pytest.raises(NotImplementedError, match="step"):
            expr_pybamm.to_rust(g, {})

    def test_index_over_negative_start_clamps(self):
        v = pybamm.Vector(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        expr_pybamm = pybamm.Index(v, slice(-7, 5))
        g = ExprGraph()
        rust_expr = expr_pybamm.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        expected = expr_pybamm.evaluate().flatten()
        assert_array_almost_equal(result, expected)


class TestInterpolantToRust:
    def test_1d_linear(self):
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
        # The first segment extrapolates, so it may not have zero width
        x = np.array([0.0, 0.0, 1.0])
        interp = pybamm.Interpolant(x, np.array([1.0, 2.0, 3.0]), pybamm.t)
        with pytest.raises(pybamm.ModelError, match=r"repeat.*convert_to_format"):
            interp.to_rust(ExprGraph(), {})

    def test_spline_above_cubic_raises_model_error(self, monkeypatch):
        from scipy import interpolate

        # scipy's cubic and pchip splines never exceed degree 3, so fake one
        monkeypatch.setattr(
            interpolate,
            "CubicSpline",
            lambda x, y, extrapolate: interpolate.PPoly(np.ones((5, x.size - 1)), x),
        )
        x = np.linspace(0.0, 1.0, 5)
        interp = pybamm.Interpolant(
            x, x**2, pybamm.StateVector(slice(0, 1)), interpolator="cubic"
        )
        with pytest.raises(pybamm.ModelError, match=r"degree 4.*convert_to_format"):
            interp.to_rust(ExprGraph(), {})

    def _check_interp_parity(self, interpolator):
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
        """Rust ignores extrapolate=False, extending the data as evaluate() does
        with extrapolate=True: a NaN would poison the solver residual."""

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


class TestConcatenationToRust:
    def test_numpy_concatenation(self):
        v1 = pybamm.Vector(np.array([1.0, 2.0]))
        v2 = pybamm.Vector(np.array([3.0, 4.0, 5.0]))
        expr_pybamm = pybamm.NumpyConcatenation(v1, v2)
        g = ExprGraph()
        rust_symbols = {}
        rust_expr = expr_pybamm.to_rust(g, rust_symbols)
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        expected = expr_pybamm.evaluate().flatten()
        assert_array_almost_equal(result, expected)

    def test_domain_concatenation_repeated_over_a_secondary_domain(
        self, dual_backend_compare
    ):

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
        dual_backend_compare(expr, y=y)


class TestVectorFieldToRust:
    def test_vector_field(self):
        v1 = pybamm.Vector(np.array([1.0, 2.0]))
        v2 = pybamm.Vector(np.array([3.0, 4.0]))
        expr_pybamm = pybamm.VectorField(v1, v2)
        g = ExprGraph()
        rust_expr = expr_pybamm.to_rust(g, {})
        result = g.eval_to_array(rust_expr, 0.0, np.array([]), np.array([]), [])
        assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])


# Distinct, positive states away from every kink, tie and interpolation knot below
_Y = np.array([0.3, 1.2, 0.7, 2.1, 0.9, 1.6])


def _parity_cases():
    """Lowered nodes on states or inputs, so constant folding cannot remove them,
    each with the points to check it at."""

    def y(start, stop):
        return pybamm.StateVector(slice(start, stop))

    a = pybamm.InputParameter("a")
    b = pybamm.InputParameter("b")
    left, right = y(0, 3), y(3, 6)
    at_y = [{"y": _Y}]
    ordered = [{"inputs": {"a": a_val, "b": 5.0}} for a_val in (3.0, 5.0, 7.0)]
    x_data = np.linspace(0.0, 3.0, 13)
    # Uneven grids, so a transposed coefficient layout cannot cancel out
    grid = (
        np.array([0.0, 0.5, 1.3, 2.0, 3.1, 4.0]),
        np.array([-1.0, -0.3, 0.4, 1.0, 2.2]),
        np.array([0.0, 1.0, 2.5, 3.5, 5.0]),
    )
    mesh_2d = np.meshgrid(*grid[:2], indexing="ij")
    mesh_3d = np.meshgrid(*grid, indexing="ij")
    time_data = pybamm.DiscreteTimeData(
        np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), "data"
    )
    cases = {
        "addition": (left + right, at_y),
        "subtraction": (left - right, at_y),
        "multiplication": (left * right, at_y),
        "division": (left / right, at_y),
        "power": (left**right, at_y),
        "scalar_operands": (2.0 * left + 1.0, at_y),
        "inner": (pybamm.Inner(left, right), at_y),
        "minimum": (pybamm.minimum(left, right), at_y),
        "maximum": (pybamm.maximum(left, right), at_y),
        "modulo": (pybamm.Modulo(a, b), [{"inputs": {"a": 10.0, "b": 3.0}}]),
        "hypot": (pybamm.hypot(left, right), at_y),
        "equal_heaviside": (pybamm.EqualHeaviside(a, b), ordered),
        "not_equal_heaviside": (pybamm.NotEqualHeaviside(a, b), ordered),
        "dense_matmul": (
            pybamm.Matrix(np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]])) @ left,
            at_y,
        ),
        "sparse_matmul": (
            pybamm.Matrix(csr_matrix(([1.0, 2.0], ([0, 1], [0, 2])), shape=(2, 3)))
            @ left,
            at_y,
        ),
        "sparse_stack": (
            pybamm.SparseStack(
                pybamm.Matrix(csr_matrix(np.array([[1.0], [0.0]]))),
                pybamm.Matrix(csr_matrix(np.array([[0.0], [3.0], [2.0]]))),
            ),
            [{}],
        ),
        "negate": (-left, at_y),
        "absolute_value": (abs(left - 1.0), at_y),
        "sign": (
            pybamm.sign(left - 1.0),
            [{"y": _Y}, {"y": np.array([1.0, 0.3, 1.7, 2.1, 0.9, 1.6])}],
        ),
        "floor": (pybamm.Floor(3 * left), at_y),
        "ceiling": (pybamm.Ceiling(3 * left), at_y),
        "index": (pybamm.Index(y(0, 6), slice(2, 5)), at_y),
        "index_negative_stop": (pybamm.Index(y(0, 6), slice(1, -1)), at_y),
        "not_constant": (pybamm.NotConstant(left), at_y),
        "explicit_time_integral": (
            pybamm.ExplicitTimeIntegral(left, pybamm.Scalar(0.0)),
            at_y,
        ),
        "discrete_time_sum": (
            pybamm.DiscreteTimeSum((time_data - y(0, 1)) ** 2),
            [{"t": 0.5, "y": _Y}],
        ),
        "max": (pybamm.max(y(0, 6)), at_y),
        "min": (pybamm.min(y(0, 6)), at_y),
        "arcsinh2": (pybamm.arcsinh2(left - 1.0, right), at_y),
        "reg_power": (pybamm.reg_power(left - 1.0, 0.7), at_y),
        "state_vector_multi_slice": (
            pybamm.StateVector(slice(0, 2), slice(4, 6)),
            at_y,
        ),
        "state_vector_dot": (
            pybamm.StateVectorDot(slice(1, 4)),
            [{"y_dot": np.arange(10.0, 16.0)}],
        ),
        "state_vector_dot_multi_slice": (
            pybamm.StateVectorDot(slice(0, 2), slice(4, 6)),
            [{"y_dot": np.arange(10.0, 16.0)}],
        ),
        "time": (
            pybamm.sqrt(y(0, 1) ** 2 + 1.0) * pybamm.exp(-pybamm.t),
            [{"t": 2.0, "y": _Y}],
        ),
        "inputs": (
            pybamm.NumpyConcatenation(
                a * y(0, 1) * y(1, 2) + pybamm.t,
                pybamm.sin(y(0, 1)) * b + pybamm.exp(-y(1, 2)),
            ),
            [{"t": 0.7, "y": _Y, "inputs": {"a": 2.5, "b": -0.8}}],
        ),
        "numpy_concatenation": (pybamm.NumpyConcatenation(left, 2 * right), at_y),
        # VectorField has no evaluate()
        "vector_field": (
            pybamm.VectorField(y(0, 2), 2 * y(2, 4), y(0, 2) + y(2, 4)),
            [{"y": _Y, "check_evaluate": False}],
        ),
        "conditional": (
            pybamm.Conditional(
                pybamm.InputParameter("s"), 2 * a, b + 5.0, pybamm.Scalar(42.0)
            ),
            # Half-integer selectors fall between the branches to the default
            [
                {"inputs": {"s": s, "a": 10.0, "b": 20.0}}
                for s in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
            ],
        ),
    }
    for name in (
        "sqrt",
        "exp",
        "log",
        "sin",
        "cos",
        "tanh",
        "sinh",
        "cosh",
        "arcsinh",
        "arctan",
        "erf",
    ):
        cases[name] = (getattr(pybamm, name)(y(0, 6)), at_y)
    for interpolator in ("linear", "cubic", "pchip"):
        cases[f"interpolant_{interpolator}"] = (
            pybamm.Interpolant(
                x_data, np.sin(x_data) + x_data, y(0, 6), interpolator=interpolator
            ),
            at_y,
        )
        cases[f"interpolant_{interpolator}_vector_valued"] = (
            pybamm.Interpolant(
                x_data,
                np.column_stack([np.sin(x_data), x_data**2]),
                y(0, 1),
                interpolator=interpolator,
            ),
            at_y,
        )
    # CasADi's 2D and 3D cubic disagree with evaluate(), so only linear is compared
    cases["interpolant_2d_linear"] = (
        pybamm.Interpolant(
            grid[:2],
            np.sin(mesh_2d[0]) * np.exp(-0.3 * mesh_2d[1]),
            (y(0, 1), y(1, 2)),
        ),
        [{"y": _Y}],
    )
    cases["interpolant_3d_linear"] = (
        pybamm.Interpolant(
            grid,
            np.cos(mesh_3d[0]) * mesh_3d[1] + np.sqrt(1 + mesh_3d[2]),
            (y(0, 1), y(1, 2), y(2, 3)),
        ),
        [{"y": _Y}],
    )
    return cases


class TestRustCasadiParity:
    @pytest.mark.parametrize("case", list(_parity_cases()))
    def test_matches_evaluate_and_casadi(self, case, dual_backend_compare):
        expr, points = _parity_cases()[case]
        for point in points:
            dual_backend_compare(expr, **point)

    @pytest.mark.parametrize(
        ("a", "b"), [(-1.0, 3.0), (1.0, -3.0), (-7.5, 2.0), (-7.0, -3.0)]
    )
    def test_modulo_with_negative_operands_truncates_like_casadi(
        self, a, b, dual_backend_compare
    ):
        # evaluate() takes the floored remainder; Rust keeps CasADi's fmod

        expr = pybamm.Modulo(pybamm.InputParameter("a"), pybamm.InputParameter("b"))
        dual_backend_compare(expr, inputs={"a": a, "b": b}, check_evaluate=False)

    @pytest.mark.parametrize(("a", "b"), [(3.0, 3.0), (3.0, 4.0)])
    def test_equality_matches_evaluate(self, a, b):
        # Equality has no CasADi form, so evaluate() is the only reference

        expr = pybamm.Equality(pybamm.InputParameter("a"), pybamm.InputParameter("b"))
        inputs = {"a": a, "b": b}
        graph = ExprGraph()
        compiled = graph.compile(expr.to_rust(graph, {}))
        np.testing.assert_array_equal(
            compiled(0.0, np.array([]), inputs), np.ravel(expr.evaluate(inputs=inputs))
        )


class TestSparseMatrixToRust:
    @pytest.mark.parametrize(
        "entries",
        [
            csc_matrix((1, 1)),
            csr_matrix(([2.0], ([1], [0])), shape=(3, 1)),
        ],
        ids=["all-zero", "column"],
    )
    def test_sparse_matrix_as_value(self, entries):
        g = ExprGraph()
        expr = pybamm.Matrix(entries).to_rust(g)
        result = g.eval_to_array(expr, 0.0, np.array([]), np.array([]), [])
        np.testing.assert_array_equal(result, entries.toarray().ravel())

    def test_sparse_matrix_in_elementwise_product(self):
        mat = pybamm.Matrix(csr_matrix(([2.0], ([1], [0])), shape=(3, 1)))
        expr = mat * pybamm.StateVector(slice(0, 1))
        g = ExprGraph()
        result = g.eval_to_array(
            expr.to_rust(g), 0.0, np.array([3.0]), np.array([]), []
        )
        np.testing.assert_array_equal(result, [0.0, 6.0, 0.0])

    def test_sparse_matrix_as_matmul_operand_and_value(self):
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


class TestRustUnsupportedNodeErrors:
    def test_unsupported_4d_interpolant_message(self):
        x = tuple(np.linspace(0, 1, 5) for _ in range(4))
        z = np.zeros((5, 5, 5, 5))
        svs = tuple(pybamm.StateVector(slice(i, i + 1)) for i in range(4))
        interp = pybamm.Interpolant(x, z, svs, interpolator="linear")
        with pytest.raises(NotImplementedError, match=r"convert_to_format"):
            interp.to_rust(ExprGraph(), {})

    def test_matmul_with_a_stacked_left_operand_message(self):
        # Only an Array lowers to a matrix node; a stack of them stays a value
        stack = pybamm.SparseStack(
            pybamm.Matrix(np.array([[1.0, 0.0, 2.0]])),
            pybamm.Matrix(np.array([[0.0, 3.0, 0.0]])),
        )
        graph = ExprGraph()
        expr = (stack @ pybamm.StateVector(slice(0, 3))).to_rust(graph, {})
        with pytest.raises(
            NotImplementedError, match=r"constant matrix.*convert_to_format"
        ):
            graph.compile(expr, n_states=3)

    @pytest.mark.parametrize(
        "make_expr",
        [
            lambda matrix: matrix,
            lambda matrix: (
                pybamm.SparseStack(matrix, matrix) @ pybamm.StateVector(slice(0, 3))
            ),
        ],
        ids=["value", "stacked-matmul-operand"],
    )
    def test_sparse_matrix_outside_a_matmul_raises(self, make_expr):
        # Refused at lowering, before a dense rows * cols copy is made
        matrix = pybamm.Matrix(csr_matrix(np.array([[1.0, 0.0, 2.0]])), name="stencil")
        with pytest.raises(
            NotImplementedError,
            match=r"1x3 sparse matrix 'stencil'.*convert_to_format",
        ):
            make_expr(matrix).to_rust(ExprGraph(), {})

    def test_unsupported_base_symbol_message(self):
        sym = pybamm.SpatialVariable("x", domain=["negative electrode"])
        with pytest.raises(NotImplementedError, match=r"convert_to_format") as exc:
            sym.to_rust(ExprGraph(), {})
        # The fix it suggests is a model setting, not a solver argument
        assert "evaluator=" not in str(exc.value)

    def test_unsupported_generic_function_message(self):
        sv = pybamm.StateVector(slice(0, 1))
        fun = pybamm.Function(np.sin, sv)
        with pytest.raises(NotImplementedError, match=r"convert_to_format"):
            fun.to_rust(ExprGraph(), {})

    def test_specific_function_without_a_rust_form_message(self):
        fun = pybamm.SpecificFunction(np.abs, pybamm.StateVector(slice(0, 1)))
        with pytest.raises(NotImplementedError, match=r"no native Rust implementation"):
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
        with pytest.raises(NotImplementedError, match=rf"'{name}'.*convert_to_format"):
            nodes[name].to_rust(ExprGraph(), {})


class TestCrossGraphGuard:
    def test_builder_rejects_foreign_expr(self):
        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        b = g_b.scalar(3.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            g_a.add(a, b)

    def test_dunder_rejects_foreign_expr(self):
        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        b = g_b.scalar(3.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            _ = a + b

    def test_compile_rejects_foreign_expr(self):
        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            g_b.compile(a)

    def test_compile_outputs_rejects_foreign_expr(self):
        g_a = ExprGraph()
        g_b = ExprGraph()
        a = g_a.scalar(2.0)
        with pytest.raises(ValueError, match="different ExprGraph"):
            g_b.compile_outputs({"x": a})


class TestRustSubgradientParity:
    def _rust_jac_row(self, expr_pybamm, n, y):
        g = ExprGraph()
        expr = expr_pybamm.to_rust(g, {})
        f = g.compile(expr, n_states=n)
        return f.jacobian("y")(0.0, y, np.array([])).toarray()

    def test_max_jacobian_matches_casadi(self):
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

        y = np.array([0.3, 0.5, 0.5, 0.3])
        if reduction == "min":
            y = -y
        sv = pybamm.StateVector(slice(0, 4))
        rust_row = self._rust_jac_row(getattr(pybamm, reduction)(sv), 4, y)
        np.testing.assert_array_equal(rust_row, [[0.0, 1.0, 0.0, 0.0]])
