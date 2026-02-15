#
# Tests for the Function classes
#

import itertools

import numpy as np
import pytest
import sympy
from scipy import special

import pybamm
from tests import (
    function_test,
    multi_var_function_test,
)


class TestFunction:
    def test_number_input(self):
        # with numbers
        log = pybamm.Function(np.log, 10)
        assert isinstance(log.children[0], pybamm.Scalar)
        assert log.evaluate() == np.log(10)

        summ = pybamm.Function(multi_var_function_test, 1, 2)
        assert isinstance(summ.children[0], pybamm.Scalar)
        assert isinstance(summ.children[1], pybamm.Scalar)
        assert summ.evaluate() == 3

    def test_function_of_one_variable(self):
        a = pybamm.Symbol("a")
        funca = pybamm.Function(function_test, a)
        assert funca.name == "function (function_test)"
        assert str(funca) == "function_test(a)"
        assert funca.children[0].name == a.name

        b = pybamm.Scalar(1)
        sina = pybamm.Function(np.sin, b)
        assert sina.evaluate() == np.sin(1)
        assert sina.name == f"function ({np.sin.__name__})"

        c = pybamm.Vector(np.linspace(0, 1))
        cosb = pybamm.Function(np.cos, c)
        np.testing.assert_array_equal(cosb.evaluate(), np.cos(c.evaluate()))

        var = pybamm.StateVector(slice(0, 100))
        y = np.linspace(0, 1, 100)[:, np.newaxis]
        logvar = pybamm.Function(np.log1p, var)
        np.testing.assert_array_equal(logvar.evaluate(y=y), np.log1p(y))

    def test_diff(self):
        a = pybamm.StateVector(slice(0, 1))
        func = pybamm.Function(function_test, a)
        with pytest.raises(
            NotImplementedError,
            match=r"Derivative of base Function class is not implemented",
        ):
            func.diff(a)

    def test_exceptions(self):
        a = pybamm.Variable("a", domain="something")
        b = pybamm.Variable("b", domain="something else")
        with pytest.raises(pybamm.DomainError):
            pybamm.Function(multi_var_function_test, a, b)

    def test_function_unnamed(self):
        fun = pybamm.Function(np.cos, pybamm.t)
        assert fun.name == "function (cos)"

    def test_to_equation(self):
        a = pybamm.Symbol("a", domain="test")

        # Test print_name
        func = pybamm.Arcsinh(a)
        func.print_name = "test"
        assert func.to_equation() == sympy.Symbol("test")

        # Test Arcsinh
        assert pybamm.Arcsinh(a).to_equation() == sympy.asinh("a")

        # Test Arctan
        assert pybamm.Arctan(a).to_equation() == sympy.atan("a")

        # Test Exp
        assert pybamm.Exp(a).to_equation() == sympy.exp("a")

        # Test log
        value = 54.0
        assert pybamm.Log(value).to_equation() == sympy.log(value)

        # Test sinh
        assert pybamm.Sinh(a).to_equation() == sympy.sinh("a")

        # Test Function
        value = 10
        assert pybamm.Function(np.log, value).to_equation() == value

    def test_to_from_json_error(self):
        a = pybamm.Symbol("a")
        funca = pybamm.Function(function_test, a)

        with pytest.raises(NotImplementedError):
            funca.to_json()

        with pytest.raises(NotImplementedError):
            pybamm.Function._from_json({})

    def test_arcsinh2(self, mocker):
        """Test arcsinh2(a, b) = arcsinh(a/b) with regularisation.

        The key feature is that arcsinh2 returns FINITE values for ALL inputs,
        including a=0, b=0, and a=b=0.
        """
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        fun = pybamm.arcsinh2(a, b)
        da_fun = fun.diff(a)
        db_fun = fun.diff(b)
        assert isinstance(fun, pybamm.Arcsinh2)

        fun_true = pybamm.arcsinh(a / b)
        da_fun_true = fun_true.diff(a)
        db_fun_true = fun_true.diff(b)

        # Make sure arcsinh2 returns finite values and matches arcsinh(a/b),
        # except when b is zero
        test_values = 10.0 ** np.linspace(-10, 3, 10)
        test_values = np.concatenate([-test_values, [0], test_values])
        test_values.sort()

        for a_val, b_val in itertools.product(test_values, test_values):
            result = fun.evaluate(inputs={"a": a_val, "b": b_val})
            assert np.isfinite(result), (
                f"arcsinh2({a_val}, {b_val}) returned non-finite: {result}"
            )

            # Arcsinh2 differentiation
            da_2 = da_fun.evaluate(inputs={"a": a_val, "b": b_val})
            db_2 = db_fun.evaluate(inputs={"a": a_val, "b": b_val})

            assert np.isfinite(da_2)
            assert np.isfinite(db_2)

            # Compare witb the true arcsinh function for non-zero b
            if b_val == 0:
                continue

            expected = np.arcsinh(a_val / b_val)
            assert result == pytest.approx(expected, rel=1e-9)

            # Arcsinh analytical derivatives
            da_true = da_fun_true.evaluate(inputs={"a": a_val, "b": b_val})
            db_true = db_fun_true.evaluate(inputs={"a": a_val, "b": b_val})

            assert da_2 == pytest.approx(da_true, rel=1e-9)
            assert db_2 == pytest.approx(db_true, rel=1e-9)

        # Test serialisation and epsilon
        eps = 1e-10
        fun_custom_eps = pybamm.arcsinh2(a, b, eps=eps)
        assert fun_custom_eps.eps == eps

        # Test new_copy preserves epsilon
        new_copy = fun_custom_eps._function_new_copy([a, b])
        assert new_copy.eps == eps

        # Test sympy conversion
        sym_a = sympy.Symbol("a")
        sym_b = sympy.Symbol("b")
        sympy_expr = fun._sympy_operator(sym_a, sym_b)
        assert sympy_expr is not None

        # Test derivative at a=0 is non-zero (critical for Newton solver convergence)
        # d/da[arcsinh(a/b)] at a=0 should be 1/b, not 0
        for b_val in [1.0, 0.1, 0.01]:
            da_at_zero = da_fun.evaluate(inputs={"a": 0.0, "b": b_val})
            expected_da = 1.0 / b_val  # d/da[arcsinh(a/b)] = 1/b at a=0
            assert da_at_zero == pytest.approx(expected_da, rel=1e-6), (
                f"d(arcsinh2)/da at a=0, b={b_val} should be {expected_da}, got {da_at_zero}"
            )

        # Test at b=0 boundary (uses regularization with eps)
        # When b=0, b_eff = eps, so derivative should be finite
        eps = fun.eps
        da_at_b0 = da_fun.evaluate(inputs={"a": 1.0, "b": 0.0})
        assert np.isfinite(da_at_b0), "d(arcsinh2)/da at b=0 should be finite"
        # Expected: sign(0) = 1, so da = 1 / hypot(a, eps) ≈ 1 for small eps
        assert da_at_b0 == pytest.approx(1.0, rel=1e-6)

        # Test arcsinh2 value at b=0
        f_at_b0 = fun.evaluate(inputs={"a": 1.0, "b": 0.0})
        assert np.isfinite(f_at_b0), "arcsinh2(1, 0) should be finite"
        # arcsinh(1/eps) is large but finite
        assert f_at_b0 > 10  # Should be arcsinh(1/1e-16) which is huge

        # Test to_json
        json_repr = fun.to_json()
        assert json_repr["function"] == "arcsinh2"
        assert "eps" in json_repr

        input_json = {
            "name": "arcsinh2",
            "id": mocker.ANY,
            "function": "arcsinh2",
            "children": [a, b],
            "eps": eps,
        }
        assert pybamm.Arcsinh2._from_json(input_json) == fun

    def test_reg_pow(self):
        x = pybamm.InputParameter("x")
        delta = pybamm.settings.tolerances.get("reg_power", 1e-3)

        # Test multiple exponents
        for a in [0.5, 1 / 3, 0.25, 0.75]:
            expr = pybamm.reg_power(x, a)
            deriv = expr.diff(x)

            # Test over full range including zero and negative values
            test_values = np.concatenate(
                [
                    -(10.0 ** np.linspace(3, -10, 50)),
                    [0],
                    10.0 ** np.linspace(-10, 3, 50),
                ]
            )

            for x_val in test_values:
                result = expr.evaluate(inputs={"x": x_val})
                deriv_result = deriv.evaluate(inputs={"x": x_val})

                # Must be finite for ALL inputs
                assert np.isfinite(result), f"reg_power({x_val}, {a}) is not finite"
                assert np.isfinite(deriv_result), (
                    f"d/dx reg_power({x_val}, {a}) is not finite"
                )

                # Check anti-symmetry: reg_power(-x, a) = -reg_power(x, a)
                result_neg = expr.evaluate(inputs={"x": -x_val})
                assert result_neg == pytest.approx(-result, rel=1e-12)

                # For large |x|, should approach |x|^a * sign(x)
                if abs(x_val) > 1000 * delta:
                    expected = np.sign(x_val) * abs(x_val) ** a
                    assert result == pytest.approx(expected, rel=1e-5)

        # Test scale parameter
        scale = 10.0
        expr_scaled = pybamm.reg_power(x, 0.5, scale=scale)
        # reg_power(x, a, scale=s) = reg_power(x/s, a) * s^a
        # For large x: should approach sqrt(x)
        assert expr_scaled.evaluate(inputs={"x": 100.0}) == pytest.approx(
            10.0, rel=1e-2
        )

        # Test that result is a RegPower instance with scale as third child
        assert isinstance(expr_scaled, pybamm.RegPower)
        # Scale is the third child
        assert expr_scaled.children[2] == pybamm.Scalar(scale)

        # Test differentiation with respect to varying exponent
        y = pybamm.InputParameter("y")
        rp_var_exp = pybamm.reg_power(x, y)
        drp_dy = rp_var_exp.diff(y)
        # Should be finite for all inputs
        for x_val in [2.0, 0.0, -2.0]:
            for y_val in [0.5, 0.25, 1.0]:
                result = drp_dy.evaluate(inputs={"x": x_val, "y": y_val})
                assert np.isfinite(result), (
                    f"d/dy reg_power({x_val}, {y_val}) is not finite"
                )

        # Test jacobian with varying exponent (StateVectors)
        sv_x = pybamm.StateVector(slice(0, 1))
        sv_a = pybamm.StateVector(slice(1, 2))
        rp_jac = pybamm.RegPower(sv_x, sv_a)
        jac = rp_jac.jac(pybamm.StateVector(slice(0, 2)))
        # Verify jacobian is finite even at x=0
        for y in [np.array([0.0, 0.5]), np.array([2.0, 0.5]), np.array([-2.0, 0.25])]:
            jac_result = jac.evaluate(y=y)
            # Convert sparse matrix to dense if needed
            if hasattr(jac_result, "toarray"):
                jac_result = jac_result.toarray()
            assert np.isfinite(jac_result).all(), (
                f"jacobian of reg_power at y={y} is not finite"
            )

        # Test jacobian when base is constant (exponent varies)
        const_base = pybamm.Scalar(4.0)
        sv_a_only = pybamm.StateVector(slice(0, 1))
        rp_const_base = pybamm.RegPower(const_base, sv_a_only)
        jac_const_base = rp_const_base.jac(sv_a_only)
        result = jac_const_base.evaluate(y=np.array([0.5]))
        # Convert sparse matrix to dense if needed
        if hasattr(result, "toarray"):
            result = result.toarray()
        assert np.isfinite(result).all()


class TestSpecificFunctions:
    def test_to_json(self, mocker):
        a = pybamm.InputParameter("a")
        fun = pybamm.cos(a)

        expected_json = {
            "name": "function (cos)",
            "id": mocker.ANY,
            "function": "cos",
        }

        assert fun.to_json() == expected_json

    def test_arcsinh(self, mocker):
        a = pybamm.InputParameter("a")
        fun = pybamm.arcsinh(a)
        assert isinstance(fun, pybamm.Arcsinh)
        assert fun.evaluate(inputs={"a": 3}) == np.arcsinh(3)
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.arcsinh(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

        # Test broadcast gets switched
        broad_a = pybamm.PrimaryBroadcast(a, "test")
        fun_broad = pybamm.arcsinh(broad_a)
        assert fun_broad == pybamm.PrimaryBroadcast(fun, "test")

        broad_a = pybamm.FullBroadcast(a, "test", "test2")
        fun_broad = pybamm.arcsinh(broad_a)
        assert fun_broad == pybamm.FullBroadcast(fun, "test", "test2")

        # Test recursion
        broad_a = pybamm.PrimaryBroadcast(pybamm.PrimaryBroadcast(a, "test"), "test2")
        fun_broad = pybamm.arcsinh(broad_a)
        assert fun_broad == pybamm.PrimaryBroadcast(
            pybamm.PrimaryBroadcast(fun, "test"), "test2"
        )

        # test creation from json
        input_json = {
            "name": "arcsinh",
            "id": mocker.ANY,
            "function": "arcsinh",
            "children": [a],
        }
        assert pybamm.Arcsinh._from_json(input_json) == fun

    def test_arctan(self, mocker):
        a = pybamm.InputParameter("a")
        fun = pybamm.arctan(a)
        assert isinstance(fun, pybamm.Arctan)
        assert fun.evaluate(inputs={"a": 3}) == np.arctan(3)
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.arctan(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

        # test creation from json
        input_json = {
            "name": "arctan",
            "id": mocker.ANY,
            "function": "arctan",
            "children": [a],
        }
        assert pybamm.Arctan._from_json(input_json) == fun

    def test_cos(self, mocker):
        a = pybamm.InputParameter("a")
        fun = pybamm.cos(a)
        assert isinstance(fun, pybamm.Cos)
        assert fun.children[0] == a
        assert fun.evaluate(inputs={"a": 3}) == np.cos(3)
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.cos(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

        # test creation from json
        input_json = {
            "name": "cos",
            "id": mocker.ANY,
            "function": "cos",
            "children": [a],
        }
        assert pybamm.Cos._from_json(input_json) == fun

    def test_cosh(self, mocker):
        a = pybamm.InputParameter("a")
        fun = pybamm.cosh(a)
        assert isinstance(fun, pybamm.Cosh)
        assert fun.children[0] == a
        assert fun.evaluate(inputs={"a": 3}) == np.cosh(3)
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.cosh(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

        # test creation from json
        input_json = {
            "name": "cosh",
            "id": mocker.ANY,
            "function": "cosh",
            "children": [a],
        }
        assert pybamm.Cosh._from_json(input_json) == fun

    def test_exp(self, mocker):
        a = pybamm.InputParameter("a")
        fun = pybamm.exp(a)
        assert isinstance(fun, pybamm.Exp)
        assert fun.children[0] == a
        assert fun.evaluate(inputs={"a": 3}) == np.exp(3)
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.exp(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

        # test creation from json
        input_json = {
            "name": "exp",
            "id": mocker.ANY,
            "function": "exp",
            "children": [a],
        }
        assert pybamm.Exp._from_json(input_json) == fun

    def test_log(self, mocker):
        a = pybamm.InputParameter("a")
        fun = pybamm.log(a)
        assert fun.evaluate(inputs={"a": 3}) == np.log(3)
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.log(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

        # Base 10
        fun = pybamm.log10(a)
        assert fun.evaluate(inputs={"a": 3}) == pytest.approx(np.log10(3))
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.log10(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

        # test creation from json
        a = pybamm.InputParameter("a")
        fun = pybamm.log(a)
        input_json = {
            "name": "log",
            "id": mocker.ANY,
            "function": "log",
            "children": [a],
        }
        assert pybamm.Log._from_json(input_json) == fun

    def test_max(self):
        a = pybamm.StateVector(slice(0, 3))
        y_test = np.array([1, 2, 3])
        fun = pybamm.max(a)
        assert isinstance(fun, pybamm.Function)
        assert fun.evaluate(y=y_test) == 3

    def test_min(self):
        a = pybamm.StateVector(slice(0, 3))
        y_test = np.array([1, 2, 3])
        fun = pybamm.min(a)
        assert isinstance(fun, pybamm.Function)
        assert fun.evaluate(y=y_test) == 1

    def test_sin(self, mocker):
        a = pybamm.InputParameter("a")
        fun = pybamm.sin(a)
        assert isinstance(fun, pybamm.Sin)
        assert fun.children[0] == a
        assert fun.evaluate(inputs={"a": 3}) == np.sin(3)
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.sin(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

        # test creation from json
        input_json = {
            "name": "sin",
            "id": mocker.ANY,
            "function": "sin",
            "children": [a],
        }
        assert pybamm.Sin._from_json(input_json) == fun

    def test_sinh(self, mocker):
        a = pybamm.InputParameter("a")
        fun = pybamm.sinh(a)
        assert isinstance(fun, pybamm.Sinh)
        assert fun.children[0] == a
        assert fun.evaluate(inputs={"a": 3}) == np.sinh(3)
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.sinh(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

        # test creation from json
        input_json = {
            "name": "sinh",
            "id": mocker.ANY,
            "function": "sinh",
            "children": [a],
        }
        assert pybamm.Sinh._from_json(input_json) == fun

    def test_sqrt(self, mocker):
        a = pybamm.InputParameter("a")
        fun = pybamm.sqrt(a)
        assert isinstance(fun, pybamm.Sqrt)
        assert fun.evaluate(inputs={"a": 3}) == np.sqrt(3)
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.sqrt(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

        # test creation from json
        input_json = {
            "name": "sqrt",
            "id": mocker.ANY,
            "function": "sqrt",
            "children": [a],
        }
        assert pybamm.Sqrt._from_json(input_json) == fun

    def test_tanh(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.tanh(a)
        assert fun.evaluate(inputs={"a": 3}) == np.tanh(3)
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.tanh(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

    def test_erf(self, mocker):
        a = pybamm.InputParameter("a")
        fun = pybamm.erf(a)
        assert fun.evaluate(inputs={"a": 3}) == special.erf(3)
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.erf(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )

        # test creation from json
        input_json = {
            "name": "erf",
            "id": mocker.ANY,
            "function": "erf",
            "children": [a],
        }
        assert pybamm.Erf._from_json(input_json) == fun

    def test_erfc(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.erfc(a)
        assert fun.evaluate(inputs={"a": 3}) == pytest.approx(
            special.erfc(3), abs=1e-15
        )
        h = 0.0000001
        assert fun.diff(a).evaluate(inputs={"a": 3}) == pytest.approx(
            (
                pybamm.erfc(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            abs=1e-05,
        )


class TestNonObjectFunctions:
    def test_normal_pdf(self):
        x = pybamm.InputParameter("x")
        mu = pybamm.InputParameter("mu")
        sigma = pybamm.InputParameter("sigma")
        fun = pybamm.normal_pdf(x, mu, sigma)
        assert fun.evaluate(inputs={"x": 0, "mu": 0, "sigma": 1}) == 1 / np.sqrt(
            2 * np.pi
        )
        assert (
            fun.evaluate(inputs={"x": 2, "mu": 2, "sigma": 10})
            == 1 / np.sqrt(2 * np.pi) / 10
        )
        assert fun.evaluate(inputs={"x": 100, "mu": 0, "sigma": 1}) == pytest.approx(0)
        assert fun.evaluate(inputs={"x": -100, "mu": 0, "sigma": 1}) == pytest.approx(0)
        assert fun.evaluate(inputs={"x": 1, "mu": 0, "sigma": 1}) > fun.evaluate(
            inputs={"x": 1, "mu": 0, "sigma": 2}
        )
        assert fun.evaluate(inputs={"x": -1, "mu": 0, "sigma": 1}) > fun.evaluate(
            inputs={"x": -1, "mu": 0, "sigma": 2}
        )

    def test_normal_cdf(self):
        x = pybamm.InputParameter("x")
        mu = pybamm.InputParameter("mu")
        sigma = pybamm.InputParameter("sigma")
        fun = pybamm.normal_cdf(x, mu, sigma)
        assert fun.evaluate(inputs={"x": 0, "mu": 0, "sigma": 1}) == 0.5
        assert fun.evaluate(inputs={"x": 2, "mu": 2, "sigma": 10}) == 0.5
        assert fun.evaluate(inputs={"x": 100, "mu": 0, "sigma": 1}) == pytest.approx(1)
        assert fun.evaluate(inputs={"x": -100, "mu": 0, "sigma": 1}) == pytest.approx(0)
        assert fun.evaluate(inputs={"x": 1, "mu": 0, "sigma": 1}) > fun.evaluate(
            inputs={"x": 1, "mu": 0, "sigma": 2}
        )
        assert fun.evaluate(inputs={"x": -1, "mu": 0, "sigma": 1}) < fun.evaluate(
            inputs={"x": -1, "mu": 0, "sigma": 2}
        )
