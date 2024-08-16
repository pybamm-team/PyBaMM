#
# Tests for the Function classes
#

import pytest
import unittest.mock as mock

import numpy as np
from scipy import special

import pybamm
import sympy
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
            match="Derivative of base Function class is not implemented",
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

    def test_erf(self):
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
            "id": mock.ANY,
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
