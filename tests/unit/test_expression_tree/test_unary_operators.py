#
# Tests for the Unary Operator classes
#
import pybamm

import unittest
import numpy as np
import autograd.numpy as auto_np


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

    def test_index(self):
        vec = pybamm.Vector(np.array([1, 2, 3, 4, 5]))
        # with integer
        ind = pybamm.Index(vec, 3)
        self.assertEqual(ind.evaluate(), 4)
        # with slice
        ind = pybamm.Index(vec, slice(1, 3))
        np.testing.assert_array_equal(ind.evaluate(), np.array([[2], [3]]))
        # with only stop slice
        ind = pybamm.Index(vec, slice(3))
        np.testing.assert_array_equal(ind.evaluate(), np.array([[1], [2], [3]]))

        # errors
        with self.assertRaisesRegex(TypeError, "index must be integer or slice"):
            pybamm.Index(vec, 0.0)
        with self.assertRaisesRegex(ValueError, "slice size exceeds child size"):
            pybamm.Index(vec, 5)

    def test_diff(self):
        a = pybamm.StateVector(slice(0, 1))
        y = np.array([5])

        # negation
        self.assertEqual((-a).diff(a).evaluate(y=y), -1)
        self.assertEqual((-a).diff(-a).evaluate(), 1)

        # absolute value (not implemented)
        absa = abs(a)
        with self.assertRaises(pybamm.UndefinedOperationError):
            absa.diff(a)

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

    def test_boundary_operators(self):
        a = pybamm.Symbol("a")
        boundary_a = pybamm.BoundaryOperator("boundary", a, "right")
        self.assertEqual(boundary_a.side, "right")
        self.assertEqual(boundary_a.child.id, a.id)

    def test_boundary_value(self):
        a = pybamm.Scalar(1)
        boundary_a = pybamm.boundary_value(a, "right")
        self.assertEqual(boundary_a.id, a.id)

        boundary_broad_a = pybamm.boundary_value(
            pybamm.Broadcast(a, ["negative electrode"]), "left"
        )
        self.assertEqual(boundary_broad_a.evaluate(), np.array([1]))

        a = pybamm.Symbol("a", domain=["separator"])
        boundary_a = pybamm.boundary_value(a, "right")
        self.assertIsInstance(boundary_a, pybamm.BoundaryValue)
        self.assertEqual(boundary_a.side, "right")
        self.assertEqual(boundary_a.domain, [])

    def test_average(self):
        a = pybamm.Scalar(1)
        average_a = pybamm.average(a)
        self.assertEqual(average_a.id, a.id)

        average_broad_a = pybamm.average(pybamm.Broadcast(a, ["negative electrode"]))
        self.assertEqual(average_broad_a.evaluate(), np.array([1]))

        average_conc_broad = pybamm.average(
            pybamm.Concatenation(
                pybamm.Broadcast(1, ["negative electrode"]),
                pybamm.Broadcast(2, ["separator"]),
                pybamm.Broadcast(3, ["positive electrode"]),
            )
        )
        self.assertIsInstance(average_conc_broad, pybamm.Division)

        for domain in [
            ["negative electrode"],
            ["separator"],
            ["positive electrode"],
            ["negative electrode", "separator", "positive electrode"],
        ]:
            a = pybamm.Symbol("a", domain=domain)
            x = pybamm.SpatialVariable("x", domain)
            av_a = pybamm.average(a)
            self.assertIsInstance(av_a, pybamm.Division)
            self.assertIsInstance(av_a.children[0], pybamm.Integral)
            self.assertEqual(av_a.children[0].integration_variable.domain, x.domain)
            self.assertEqual(av_a.domain, [])

        a = pybamm.Symbol("a", domain="bad domain")
        with self.assertRaises(pybamm.DomainError):
            pybamm.average(a)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
