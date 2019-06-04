#
# Tests for the Function classes
#
import pybamm

import unittest
import numpy as np
import autograd.numpy as auto_np


def test_function(arg):
    return arg + arg


def test_const_function():
    return 1


def test_multi_var_function(arg1, arg2):
    return arg1 + arg2


class TestFunction(unittest.TestCase):
    def test_constant_functions(self):
        d = pybamm.Scalar(6)
        funcd = pybamm.Function(test_const_function, d)
        self.assertEqual(funcd.evaluate(), 1)

    def test_function_of_one_variable(self):
        a = pybamm.Symbol("a")
        funca = pybamm.Function(test_function, a)
        self.assertEqual(funca.name, "function (test_function)")
        self.assertEqual(funca.children[0].name, a.name)

        b = pybamm.Scalar(1)
        sina = pybamm.Function(np.sin, b)
        self.assertEqual(sina.evaluate(), np.sin(1))
        self.assertEqual(sina.name, "function ({})".format(np.sin.__name__))

        c = pybamm.Vector(np.linspace(0, 1))
        cosb = pybamm.Function(np.cos, c)
        np.testing.assert_array_equal(cosb.evaluate(), np.cos(c.evaluate()))

        var = pybamm.StateVector(slice(0, 100))
        y = np.linspace(0, 1, 100)[:, np.newaxis]
        logvar = pybamm.Function(np.log1p, var)
        np.testing.assert_array_equal(logvar.evaluate(y=y), np.log1p(y))

    def test_function_of_multiple_variables(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        func = pybamm.Function(test_multi_var_function, a, b)
        self.assertEqual(func.name, "function (test_multi_var_function)")
        self.assertEqual(func.children[0].name, a.name)
        self.assertEqual(func.children[1].name, b.name)

    def test_with_autograd(self):
        a = pybamm.StateVector(slice(0, 1))
        y = np.array([5])
        func = pybamm.Function(test_function, a)
        self.assertEqual((func).diff(a).evaluate(y=y), 2)
        self.assertEqual((func).diff(func).evaluate(), 1)
        func = pybamm.Function(auto_np.sin, a)
        self.assertEqual(func.evaluate(y=y), np.sin(a.evaluate(y=y)))
        self.assertEqual(func.diff(a).evaluate(y=y), np.cos(a.evaluate(y=y)))
        func = pybamm.Function(auto_np.exp, a)
        self.assertEqual(func.evaluate(y=y), np.exp(a.evaluate(y=y)))
        self.assertEqual(func.diff(a).evaluate(y=y), np.exp(a.evaluate(y=y)))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
