#
# Tests for the Unary Operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest
import numpy as np
import autograd.numpy as auto_np


def test_function(arg):
    return arg + arg


def test_const_function():
    return 1


class TestUnaryOperators(unittest.TestCase):
    def test_unary_operator(self):
        a = pybamm.Symbol("a", domain=["test"])
        un = pybamm.UnaryOperator("unary test", a)
        self.assertEqual(un.children[0].name, a.name)
        self.assertEqual(un.domain, a.domain)

    def test_negation(self):
        a = pybamm.Symbol("a")
        nega = pybamm.Negate(a)
        self.assertEqual(nega.name, "-")
        self.assertEqual(nega.children[0].name, a.name)

        b = pybamm.Scalar(4)
        negb = pybamm.Negate(b)
        self.assertEqual(negb.evaluate(), -4)

    def test_absolute(self):
        a = pybamm.Symbol("a")
        absa = pybamm.AbsoluteValue(a)
        self.assertEqual(absa.name, "abs")
        self.assertEqual(absa.children[0].name, a.name)

        b = pybamm.Scalar(-4)
        absb = pybamm.AbsoluteValue(b)
        self.assertEqual(absb.evaluate(), 4)

    def test_function(self):
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
        y = np.linspace(0, 1, 100)
        logvar = pybamm.Function(np.log1p, var)
        np.testing.assert_array_equal(logvar.evaluate(y=y), np.log1p(y))

        d = pybamm.Scalar(6)
        funcd = pybamm.Function(test_const_function, d)
        self.assertEqual(funcd.evaluate(), 1)

    def test_function_simplify(self):
        a = pybamm.Symbol("a")
        funca = pybamm.Function(test_const_function, a).simplify()
        self.assertIsInstance(funca, pybamm.Scalar)
        self.assertEqual(funca.evaluate(), 1)

    def test_gradient(self):
        a = pybamm.Symbol("a")
        grad = pybamm.Gradient(a)
        self.assertEqual(grad.children[0].name, a.name)

    def test_integral(self):
        # time integral
        a = pybamm.Symbol("a")
        t = pybamm.t
        inta = pybamm.Integral(a, t)
        self.assertEqual(inta.name, "integral dtime")
        # self.assertTrue(inta.definite)
        self.assertEqual(inta.children[0].name, a.name)
        self.assertEqual(inta.integration_variable, t)
        self.assertEqual(inta.domain, [])

        # space integral
        a = pybamm.Symbol("a", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", ["negative electrode"])
        inta = pybamm.Integral(a, x)
        self.assertEqual(inta.name, "integral dx ['negative electrode']")
        self.assertEqual(inta.children[0].name, a.name)
        self.assertEqual(inta.integration_variable, x)
        self.assertEqual(inta.domain, [])

        # Indefinite
        inta = pybamm.IndefiniteIntegral(a, x)
        self.assertEqual(inta.name, "a integrated w.r.t x on ['negative electrode']")
        self.assertEqual(inta.children[0].name, a.name)
        self.assertEqual(inta.integration_variable, x)
        self.assertEqual(inta.domain, ["negative electrode"])

        # expected errors
        a = pybamm.Symbol("a", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", ["separator"])
        y = pybamm.Variable("y")
        with self.assertRaises(pybamm.DomainError):
            pybamm.Integral(a, x)
        with self.assertRaises(ValueError):
            pybamm.Integral(a, y)

    def test_diff(self):
        a = pybamm.StateVector(slice(0, 1))
        y = np.array([5])

        # negation
        self.assertEqual((-a).diff(a).evaluate(y=y), -1)
        self.assertEqual((-a).diff(-a).evaluate(), 1)

        # absolute value (not implemented)
        absa = abs(a)
        with self.assertRaises(NotImplementedError):
            absa.diff(a)

        # function: use autograd
        func = pybamm.Function(test_function, a)
        self.assertEqual((func).diff(a).evaluate(y=y), 2)
        self.assertEqual((func).diff(func).evaluate(), 1)
        func = pybamm.Function(auto_np.sin, a)
        self.assertEqual(func.evaluate(y=y), np.sin(a.evaluate(y=y)))
        self.assertEqual(func.diff(a).evaluate(y=y), np.cos(a.evaluate(y=y)))
        func = pybamm.Function(auto_np.exp, a)
        self.assertEqual(func.evaluate(y=y), np.exp(a.evaluate(y=y)))
        self.assertEqual(func.diff(a).evaluate(y=y), np.exp(a.evaluate(y=y)))

        # spatial operator (not implemented)
        spatial_a = pybamm.SpatialOperator("name", a)
        with self.assertRaises(NotImplementedError):
            spatial_a.diff(a)

    def test_printing(self):
        a = pybamm.Symbol("a")
        self.assertEqual(str(-a), "-a")
        grad = pybamm.Gradient(a)
        self.assertEqual(grad.name, "grad")
        self.assertEqual(str(grad), "grad(a)")

    def test_id(self):
        a = pybamm.Scalar(4)
        un1 = pybamm.UnaryOperator("test", a)
        un2 = pybamm.UnaryOperator("test", a)
        un3 = pybamm.UnaryOperator("new test", a)
        self.assertEqual(un1.id, un2.id)
        self.assertNotEqual(un1.id, un3.id)
        a = pybamm.Scalar(4)
        un4 = pybamm.UnaryOperator("test", a)
        self.assertEqual(un1.id, un4.id)
        d = pybamm.Scalar(42)
        un5 = pybamm.UnaryOperator("test", d)
        self.assertNotEqual(un1.id, un5.id)

    def test_boundary_value(self):
        a = pybamm.Symbol("a")
        boundary_a = pybamm.boundary_value(a, "right")
        self.assertEqual(boundary_a, a)

        boundary_broad_a = pybamm.boundary_value(
            pybamm.Broadcast(a, ["negative electrode"]), "left"
        )
        self.assertEqual(boundary_broad_a.id, a.id)

        a = pybamm.Symbol("a", domain=["separator"])
        boundary_a = pybamm.boundary_value(a, "right")
        self.assertIsInstance(boundary_a, pybamm.BoundaryValue)
        self.assertEqual(boundary_a.side, "right")
        self.assertEqual(boundary_a.domain, [])

    def test_average(self):
        a = pybamm.Symbol("a")
        average_a = pybamm.average(a)
        self.assertEqual(average_a, a)

        average_broad_a = pybamm.average(pybamm.Broadcast(a, ["negative electrode"]))
        self.assertEqual(average_broad_a.id, a.id)

        for domain in [["negative electrode"], ["separator"], ["positive electrode"]]:
            a = pybamm.Symbol("a", domain=domain)
            x = pybamm.SpatialVariable("x", domain)
            av_a = pybamm.average(a)
            self.assertIsInstance(av_a, pybamm.Division)
            self.assertIsInstance(av_a.children[0], pybamm.Integral)
            self.assertEqual(av_a.children[0].integration_variable.domain, x.domain)
            self.assertEqual(av_a.domain, [])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
