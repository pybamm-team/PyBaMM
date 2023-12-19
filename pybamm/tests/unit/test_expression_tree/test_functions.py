#
# Tests for the Function classes
#
from tests import TestCase
import unittest
import unittest.mock as mock

import numpy as np
from scipy import special

import pybamm
from pybamm.util import have_optional_dependency


def test_function(arg):
    return arg + arg


def test_multi_var_function(arg1, arg2):
    return arg1 + arg2


def test_multi_var_function_cube(arg1, arg2):
    return arg1 + arg2**3


class TestFunction(TestCase):
    def test_number_input(self):
        # with numbers
        log = pybamm.Function(np.log, 10)
        self.assertIsInstance(log.children[0], pybamm.Scalar)
        self.assertEqual(log.evaluate(), np.log(10))

        summ = pybamm.Function(test_multi_var_function, 1, 2)
        self.assertIsInstance(summ.children[0], pybamm.Scalar)
        self.assertIsInstance(summ.children[1], pybamm.Scalar)
        self.assertEqual(summ.evaluate(), 3)

    def test_function_of_one_variable(self):
        a = pybamm.Symbol("a")
        funca = pybamm.Function(test_function, a)
        self.assertEqual(funca.name, "function (test_function)")
        self.assertEqual(str(funca), "test_function(a)")
        self.assertEqual(funca.children[0].name, a.name)

        b = pybamm.Scalar(1)
        sina = pybamm.Function(np.sin, b)
        self.assertEqual(sina.evaluate(), np.sin(1))
        self.assertEqual(sina.name, f"function ({np.sin.__name__})")

        c = pybamm.Vector(np.linspace(0, 1))
        cosb = pybamm.Function(np.cos, c)
        np.testing.assert_array_equal(cosb.evaluate(), np.cos(c.evaluate()))

        var = pybamm.StateVector(slice(0, 100))
        y = np.linspace(0, 1, 100)[:, np.newaxis]
        logvar = pybamm.Function(np.log1p, var)
        np.testing.assert_array_equal(logvar.evaluate(y=y), np.log1p(y))

    def test_diff(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        y = np.array([5])
        func = pybamm.Function(test_function, a)
        self.assertEqual(func.diff(a).evaluate(y=y), 2)
        self.assertEqual(func.diff(func).evaluate(), 1)
        func = pybamm.sin(a)
        self.assertEqual(func.evaluate(y=y), np.sin(a.evaluate(y=y)))
        self.assertEqual(func.diff(a).evaluate(y=y), np.cos(a.evaluate(y=y)))
        func = pybamm.exp(a)
        self.assertEqual(func.evaluate(y=y), np.exp(a.evaluate(y=y)))
        self.assertEqual(func.diff(a).evaluate(y=y), np.exp(a.evaluate(y=y)))

        # multiple variables
        func = pybamm.Function(test_multi_var_function, 4 * a, 3 * a)
        self.assertEqual(func.diff(a).evaluate(y=y), 7)
        func = pybamm.Function(test_multi_var_function, 4 * a, 3 * b)
        self.assertEqual(func.diff(a).evaluate(y=np.array([5, 6])), 4)
        self.assertEqual(func.diff(b).evaluate(y=np.array([5, 6])), 3)
        func = pybamm.Function(test_multi_var_function_cube, 4 * a, 3 * b)
        self.assertEqual(func.diff(a).evaluate(y=np.array([5, 6])), 4)
        self.assertEqual(
            func.diff(b).evaluate(y=np.array([5, 6])), 3 * 3 * (3 * 6) ** 2
        )

        # exceptions
        func = pybamm.Function(
            test_multi_var_function_cube, 4 * a, 3 * b, derivative="derivative"
        )
        with self.assertRaises(ValueError):
            func.diff(a)

    def test_function_of_multiple_variables(self):
        a = pybamm.Variable("a")
        b = pybamm.Parameter("b")
        func = pybamm.Function(test_multi_var_function, a, b)
        self.assertEqual(func.name, "function (test_multi_var_function)")
        self.assertEqual(str(func), "test_multi_var_function(a, b)")
        self.assertEqual(func.children[0].name, a.name)
        self.assertEqual(func.children[1].name, b.name)

        # test eval and diff
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        y = np.array([5, 2])
        func = pybamm.Function(test_multi_var_function, a, b)

        self.assertEqual(func.evaluate(y=y), 7)
        self.assertEqual(func.diff(a).evaluate(y=y), 1)
        self.assertEqual(func.diff(b).evaluate(y=y), 1)
        self.assertEqual(func.diff(func).evaluate(), 1)

    def test_exceptions(self):
        a = pybamm.Variable("a", domain="something")
        b = pybamm.Variable("b", domain="something else")
        with self.assertRaises(pybamm.DomainError):
            pybamm.Function(test_multi_var_function, a, b)

    def test_function_unnamed(self):
        fun = pybamm.Function(np.cos, pybamm.t)
        self.assertEqual(fun.name, "function (cos)")

    def test_to_equation(self):
        sympy = have_optional_dependency("sympy")
        a = pybamm.Symbol("a", domain="test")

        # Test print_name
        func = pybamm.Arcsinh(a)
        func.print_name = "test"
        self.assertEqual(func.to_equation(), sympy.Symbol("test"))

        # Test Arcsinh
        self.assertEqual(pybamm.Arcsinh(a).to_equation(), sympy.asinh(a))

        # Test Arctan
        self.assertEqual(pybamm.Arctan(a).to_equation(), sympy.atan(a))

        # Test Exp
        self.assertEqual(pybamm.Exp(a).to_equation(), sympy.exp(a))

        # Test log
        self.assertEqual(pybamm.Log(54.0).to_equation(), sympy.log(54.0))

        # Test sinh
        self.assertEqual(pybamm.Sinh(a).to_equation(), sympy.sinh(a))

        # Test Function
        self.assertEqual(pybamm.Function(np.log, 10).to_equation(), 10.0)

    def test_to_from_json_error(self):
        a = pybamm.Symbol("a")
        funca = pybamm.Function(test_function, a)

        with self.assertRaises(NotImplementedError):
            funca.to_json()

        with self.assertRaises(NotImplementedError):
            pybamm.Function._from_json({})


class TestSpecificFunctions(TestCase):
    def test_to_json(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.cos(a)

        expected_json = {
            "name": "function (cos)",
            "id": mock.ANY,
            "function": "cos",
        }

        self.assertEqual(fun.to_json(), expected_json)

    def test_arcsinh(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.arcsinh(a)
        self.assertIsInstance(fun, pybamm.Arcsinh)
        self.assertEqual(fun.evaluate(inputs={"a": 3}), np.arcsinh(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.arcsinh(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

        # Test broadcast gets switched
        broad_a = pybamm.PrimaryBroadcast(a, "test")
        fun_broad = pybamm.arcsinh(broad_a)
        self.assertEqual(fun_broad, pybamm.PrimaryBroadcast(fun, "test"))

        broad_a = pybamm.FullBroadcast(a, "test", "test2")
        fun_broad = pybamm.arcsinh(broad_a)
        self.assertEqual(fun_broad, pybamm.FullBroadcast(fun, "test", "test2"))

        # Test recursion
        broad_a = pybamm.PrimaryBroadcast(pybamm.PrimaryBroadcast(a, "test"), "test2")
        fun_broad = pybamm.arcsinh(broad_a)
        self.assertEqual(
            fun_broad,
            pybamm.PrimaryBroadcast(pybamm.PrimaryBroadcast(fun, "test"), "test2"),
        )

        # test creation from json
        input_json = {
            "name": "arcsinh",
            "id": mock.ANY,
            "function": "arcsinh",
            "children": [a],
        }
        self.assertEqual(pybamm.Arcsinh._from_json(input_json), fun)

    def test_arctan(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.arctan(a)
        self.assertIsInstance(fun, pybamm.Arctan)
        self.assertEqual(fun.evaluate(inputs={"a": 3}), np.arctan(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.arctan(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

        # test creation from json
        input_json = {
            "name": "arctan",
            "id": mock.ANY,
            "function": "arctan",
            "children": [a],
        }
        self.assertEqual(pybamm.Arctan._from_json(input_json), fun)

    def test_cos(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.cos(a)
        self.assertIsInstance(fun, pybamm.Cos)
        self.assertEqual(fun.children[0], a)
        self.assertEqual(fun.evaluate(inputs={"a": 3}), np.cos(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.cos(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

        # test creation from json
        input_json = {
            "name": "cos",
            "id": mock.ANY,
            "function": "cos",
            "children": [a],
        }
        self.assertEqual(pybamm.Cos._from_json(input_json), fun)

    def test_cosh(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.cosh(a)
        self.assertIsInstance(fun, pybamm.Cosh)
        self.assertEqual(fun.children[0], a)
        self.assertEqual(fun.evaluate(inputs={"a": 3}), np.cosh(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.cosh(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

        # test creation from json
        input_json = {
            "name": "cosh",
            "id": mock.ANY,
            "function": "cosh",
            "children": [a],
        }
        self.assertEqual(pybamm.Cosh._from_json(input_json), fun)

    def test_exp(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.exp(a)
        self.assertIsInstance(fun, pybamm.Exp)
        self.assertEqual(fun.children[0], a)
        self.assertEqual(fun.evaluate(inputs={"a": 3}), np.exp(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.exp(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

        # test creation from json
        input_json = {
            "name": "exp",
            "id": mock.ANY,
            "function": "exp",
            "children": [a],
        }
        self.assertEqual(pybamm.Exp._from_json(input_json), fun)

    def test_log(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.log(a)
        self.assertEqual(fun.evaluate(inputs={"a": 3}), np.log(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.log(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

        # Base 10
        fun = pybamm.log10(a)
        self.assertAlmostEqual(fun.evaluate(inputs={"a": 3}), np.log10(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.log10(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

        # test creation from json
        a = pybamm.InputParameter("a")
        fun = pybamm.log(a)
        input_json = {
            "name": "log",
            "id": mock.ANY,
            "function": "log",
            "children": [a],
        }
        self.assertEqual(pybamm.Log._from_json(input_json), fun)

    def test_max(self):
        a = pybamm.StateVector(slice(0, 3))
        y_test = np.array([1, 2, 3])
        fun = pybamm.max(a)
        self.assertIsInstance(fun, pybamm.Function)
        self.assertEqual(fun.evaluate(y=y_test), 3)

    def test_min(self):
        a = pybamm.StateVector(slice(0, 3))
        y_test = np.array([1, 2, 3])
        fun = pybamm.min(a)
        self.assertIsInstance(fun, pybamm.Function)
        self.assertEqual(fun.evaluate(y=y_test), 1)

    def test_sin(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.sin(a)
        self.assertIsInstance(fun, pybamm.Sin)
        self.assertEqual(fun.children[0], a)
        self.assertEqual(fun.evaluate(inputs={"a": 3}), np.sin(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.sin(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

        # test creation from json
        input_json = {
            "name": "sin",
            "id": mock.ANY,
            "function": "sin",
            "children": [a],
        }
        self.assertEqual(pybamm.Sin._from_json(input_json), fun)

    def test_sinh(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.sinh(a)
        self.assertIsInstance(fun, pybamm.Sinh)
        self.assertEqual(fun.children[0], a)
        self.assertEqual(fun.evaluate(inputs={"a": 3}), np.sinh(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.sinh(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

        # test creation from json
        input_json = {
            "name": "sinh",
            "id": mock.ANY,
            "function": "sinh",
            "children": [a],
        }
        self.assertEqual(pybamm.Sinh._from_json(input_json), fun)

    def test_sqrt(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.sqrt(a)
        self.assertIsInstance(fun, pybamm.Sqrt)
        self.assertEqual(fun.evaluate(inputs={"a": 3}), np.sqrt(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.sqrt(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

        # test creation from json
        input_json = {
            "name": "sqrt",
            "id": mock.ANY,
            "function": "sqrt",
            "children": [a],
        }
        self.assertEqual(pybamm.Sqrt._from_json(input_json), fun)

    def test_tanh(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.tanh(a)
        self.assertEqual(fun.evaluate(inputs={"a": 3}), np.tanh(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.tanh(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

    def test_erf(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.erf(a)
        self.assertEqual(fun.evaluate(inputs={"a": 3}), special.erf(3))
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.erf(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )

        # test creation from json
        input_json = {
            "name": "erf",
            "id": mock.ANY,
            "function": "erf",
            "children": [a],
        }
        self.assertEqual(pybamm.Erf._from_json(input_json), fun)

    def test_erfc(self):
        a = pybamm.InputParameter("a")
        fun = pybamm.erfc(a)
        self.assertAlmostEqual(
            fun.evaluate(inputs={"a": 3}), special.erfc(3), places=15
        )
        h = 0.0000001
        self.assertAlmostEqual(
            fun.diff(a).evaluate(inputs={"a": 3}),
            (
                pybamm.erfc(pybamm.Scalar(3 + h)).evaluate()
                - fun.evaluate(inputs={"a": 3})
            )
            / h,
            places=5,
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
