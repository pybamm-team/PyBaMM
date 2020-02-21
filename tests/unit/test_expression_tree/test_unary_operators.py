#
# Tests for the Unary Operator classes
#
import pybamm

import unittest
import numpy as np


class TestUnaryOperators(unittest.TestCase):
    def test_unary_operator(self):
        a = pybamm.Symbol("a", domain=["test"])
        un = pybamm.UnaryOperator("unary test", a)
        self.assertEqual(un.children[0].name, a.name)
        self.assertEqual(un.domain, a.domain)

        # with number
        log = pybamm.log(10)
        self.assertEqual(log.evaluate(), np.log(10))

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

    def test_sign(self):
        b = pybamm.Scalar(-4)
        signb = pybamm.sign(b)
        self.assertEqual(signb.evaluate(), -1)

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
        self.assertEqual(inta.integration_variable[0], t)
        self.assertEqual(inta.domain, [])

        # space integral
        a = pybamm.Symbol("a", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", ["negative electrode"])
        inta = pybamm.Integral(a, x)
        self.assertEqual(inta.name, "integral dx ['negative electrode']")
        self.assertEqual(inta.children[0].name, a.name)
        self.assertEqual(inta.integration_variable[0], x)
        self.assertEqual(inta.domain, [])
        self.assertEqual(inta.auxiliary_domains, {})
        # space integral with secondary domain
        a_sec = pybamm.Symbol(
            "a",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        x = pybamm.SpatialVariable("x", ["negative electrode"])
        inta_sec = pybamm.Integral(a_sec, x)
        self.assertEqual(inta_sec.domain, ["current collector"])
        self.assertEqual(inta_sec.auxiliary_domains, {})
        # space integral with secondary domain
        a_tert = pybamm.Symbol(
            "a",
            domain=["negative electrode"],
            auxiliary_domains={
                "secondary": "current collector",
                "tertiary": "some extra domain",
            },
        )
        x = pybamm.SpatialVariable("x", ["negative electrode"])
        inta_tert = pybamm.Integral(a_tert, x)
        self.assertEqual(inta_tert.domain, ["current collector"])
        self.assertEqual(
            inta_tert.auxiliary_domains, {"secondary": ["some extra domain"]}
        )

        # space integral over two variables
        b = pybamm.Symbol("b", domain=["current collector"])
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])
        inta = pybamm.Integral(b, [y, z])
        self.assertEqual(inta.name, "integral dy dz ['current collector']")
        self.assertEqual(inta.children[0].name, b.name)
        self.assertEqual(inta.integration_variable[0], y)
        self.assertEqual(inta.integration_variable[1], z)
        self.assertEqual(inta.domain, [])

        # Indefinite
        inta = pybamm.IndefiniteIntegral(a, x)
        self.assertEqual(inta.name, "a integrated w.r.t x on ['negative electrode']")
        self.assertEqual(inta.children[0].name, a.name)
        self.assertEqual(inta.integration_variable[0], x)
        self.assertEqual(inta.domain, ["negative electrode"])
        inta_sec = pybamm.IndefiniteIntegral(a_sec, x)
        self.assertEqual(inta_sec.domain, ["negative electrode"])
        self.assertEqual(
            inta_sec.auxiliary_domains, {"secondary": ["current collector"]}
        )

        # expected errors
        a = pybamm.Symbol("a", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", ["separator"])
        y = pybamm.Variable("y")
        z = pybamm.SpatialVariable("z", ["negative electrode"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.Integral(a, x)
        with self.assertRaises(ValueError):
            pybamm.Integral(a, y)

    def test_index(self):
        vec = pybamm.StateVector(slice(0, 5))
        y_test = np.array([1, 2, 3, 4, 5])
        # with integer
        ind = vec[3]
        self.assertIsInstance(ind, pybamm.Index)
        self.assertEqual(ind.slice, slice(3, 4))
        self.assertEqual(ind.evaluate(y=y_test), 4)
        # with slice
        ind = vec[1:3]
        self.assertIsInstance(ind, pybamm.Index)
        self.assertEqual(ind.slice, slice(1, 3))
        np.testing.assert_array_equal(ind.evaluate(y=y_test), np.array([[2], [3]]))
        # with only stop slice
        ind = vec[:3]
        self.assertIsInstance(ind, pybamm.Index)
        self.assertEqual(ind.slice, slice(3))
        np.testing.assert_array_equal(ind.evaluate(y=y_test), np.array([[1], [2], [3]]))

        # errors
        with self.assertRaisesRegex(TypeError, "index must be integer or slice"):
            pybamm.Index(vec, 0.0)
        debug_mode = pybamm.settings.debug_mode
        pybamm.settings.debug_mode = True
        with self.assertRaisesRegex(ValueError, "slice size exceeds child size"):
            pybamm.Index(vec, 5)
        pybamm.settings.debug_mode = debug_mode

    def test_diff(self):
        a = pybamm.StateVector(slice(0, 1))
        y = np.array([5])

        # negation
        self.assertEqual((-a).diff(a).evaluate(y=y), -1)
        self.assertEqual((-a).diff(-a).evaluate(), 1)

        # absolute value
        self.assertEqual((a ** 3).diff(a).evaluate(y=y), 3 * 5 ** 2)
        self.assertEqual((abs(a ** 3)).diff(a).evaluate(y=y), 3 * 5 ** 2)
        self.assertEqual((a ** 3).diff(a).evaluate(y=-y), 3 * 5 ** 2)
        self.assertEqual((abs(a ** 3)).diff(a).evaluate(y=-y), -3 * 5 ** 2)

        # sign
        self.assertEqual((pybamm.sign(a)).diff(a).evaluate(y=y), 0)

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

    def test_delta_function(self):
        a = pybamm.Symbol("a")
        delta_a = pybamm.DeltaFunction(a, "right", "some domain")
        self.assertEqual(delta_a.side, "right")
        self.assertEqual(delta_a.child.id, a.id)
        self.assertFalse(delta_a.evaluates_on_edges())
        with self.assertRaisesRegex(
            pybamm.DomainError, "Delta function domain cannot be None"
        ):
            delta_a = pybamm.DeltaFunction(a, "right", None)

    def test_boundary_operators(self):
        a = pybamm.Symbol("a", domain="some domain")
        boundary_a = pybamm.BoundaryOperator("boundary", a, "right")
        self.assertEqual(boundary_a.side, "right")
        self.assertEqual(boundary_a.child.id, a.id)

    def test_evaluates_on_edges(self):
        a = pybamm.StateVector(slice(0, 10))
        self.assertFalse(a[1].evaluates_on_edges())
        self.assertFalse(pybamm.Laplacian(a).evaluates_on_edges())

    def test_boundary_value(self):
        a = pybamm.Scalar(1)
        boundary_a = pybamm.boundary_value(a, "right")
        self.assertEqual(boundary_a.id, a.id)

        boundary_broad_a = pybamm.boundary_value(
            pybamm.PrimaryBroadcast(a, ["negative electrode"]), "left"
        )
        self.assertEqual(boundary_broad_a.evaluate(), np.array([1]))

        a = pybamm.Symbol("a", domain=["separator"])
        boundary_a = pybamm.boundary_value(a, "right")
        self.assertIsInstance(boundary_a, pybamm.BoundaryValue)
        self.assertEqual(boundary_a.side, "right")
        self.assertEqual(boundary_a.domain, [])
        self.assertEqual(boundary_a.auxiliary_domains, {})
        # test with secondary domain
        a_sec = pybamm.Symbol(
            "a",
            domain=["separator"],
            auxiliary_domains={"secondary": "current collector"},
        )
        boundary_a_sec = pybamm.boundary_value(a_sec, "right")
        self.assertEqual(boundary_a_sec.domain, ["current collector"])
        self.assertEqual(boundary_a_sec.auxiliary_domains, {})
        # test with secondary domain and tertiary domain
        a_tert = pybamm.Symbol(
            "a",
            domain=["separator"],
            auxiliary_domains={"secondary": "current collector", "tertiary": "bla"},
        )
        boundary_a_tert = pybamm.boundary_value(a_tert, "right")
        self.assertEqual(boundary_a_tert.domain, ["current collector"])
        self.assertEqual(boundary_a_tert.auxiliary_domains, {"secondary": ["bla"]})

        # error if boundary value on tabs and domain is not "current collector"
        var = pybamm.Variable("var", domain=["negative electrode"])
        with self.assertRaisesRegex(pybamm.ModelError, "Can only take boundary"):
            pybamm.boundary_value(var, "negative tab")
            pybamm.boundary_value(var, "positive tab")

    def test_average(self):
        a = pybamm.Scalar(1)
        average_a = pybamm.x_average(a)
        self.assertEqual(average_a.id, a.id)

        average_broad_a = pybamm.x_average(
            pybamm.PrimaryBroadcast(a, ["negative electrode"])
        )
        self.assertEqual(average_broad_a.evaluate(), np.array([1]))

        conc_broad = pybamm.Concatenation(
            pybamm.PrimaryBroadcast(1, ["negative electrode"]),
            pybamm.PrimaryBroadcast(2, ["separator"]),
            pybamm.PrimaryBroadcast(3, ["positive electrode"]),
        )
        average_conc_broad = pybamm.x_average(conc_broad)
        self.assertIsInstance(average_conc_broad, pybamm.Division)

        for domain in [
            ["negative electrode"],
            ["separator"],
            ["positive electrode"],
            ["negative electrode", "separator", "positive electrode"],
        ]:
            a = pybamm.Symbol("a", domain=domain)
            x = pybamm.SpatialVariable("x", domain)
            av_a = pybamm.x_average(a)
            self.assertIsInstance(av_a, pybamm.Division)
            self.assertIsInstance(av_a.children[0], pybamm.Integral)
            self.assertEqual(av_a.children[0].integration_variable[0].domain, x.domain)
            # electrode domains go to current collector when averaged
            self.assertEqual(av_a.domain, [])

        a = pybamm.Symbol("a", domain="bad domain")
        with self.assertRaises(pybamm.DomainError):
            pybamm.x_average(a)

    def test_r_average(self):
        a = pybamm.Scalar(1)
        average_a = pybamm.r_average(a)
        self.assertEqual(average_a.id, a.id)

        average_broad_a = pybamm.r_average(
            pybamm.PrimaryBroadcast(a, ["negative particle"])
        )
        self.assertEqual(average_broad_a.evaluate(), np.array([1]))

        for domain in [["negative particle"], ["positive particle"]]:
            a = pybamm.Symbol("a", domain=domain)
            r = pybamm.SpatialVariable("r", domain)
            av_a = pybamm.r_average(a)
            self.assertIsInstance(av_a, pybamm.Division)
            self.assertIsInstance(av_a.children[0], pybamm.Integral)
            self.assertEqual(av_a.children[0].integration_variable[0].domain, r.domain)
            # electrode domains go to current collector when averaged
            self.assertEqual(av_a.domain, [])

        a = pybamm.Symbol("a", domain="bad domain")
        with self.assertRaises(pybamm.DomainError):
            pybamm.x_average(a)

    def test_yz_average(self):
        a = pybamm.Scalar(1)
        z_average_a = pybamm.z_average(a)
        yz_average_a = pybamm.yz_average(a)
        self.assertEqual(z_average_a.id, a.id)
        self.assertEqual(yz_average_a.id, a.id)

        z_average_broad_a = pybamm.z_average(
            pybamm.PrimaryBroadcast(a, ["current collector"])
        )
        yz_average_broad_a = pybamm.yz_average(
            pybamm.PrimaryBroadcast(a, ["current collector"])
        )
        self.assertEqual(z_average_broad_a.evaluate(), np.array([1]))
        self.assertEqual(yz_average_broad_a.evaluate(), np.array([1]))

        a = pybamm.Symbol("a", domain=["current collector"])
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])
        z_av_a = pybamm.z_average(a)
        yz_av_a = pybamm.yz_average(a)

        self.assertIsInstance(z_av_a, pybamm.Division)
        self.assertIsInstance(yz_av_a, pybamm.Division)
        self.assertIsInstance(z_av_a.children[0], pybamm.Integral)
        self.assertIsInstance(yz_av_a.children[0], pybamm.Integral)
        self.assertEqual(z_av_a.children[0].integration_variable[0].domain, z.domain)
        self.assertEqual(yz_av_a.children[0].integration_variable[0].domain, y.domain)
        self.assertEqual(yz_av_a.children[0].integration_variable[1].domain, z.domain)
        self.assertEqual(z_av_a.domain, [])
        self.assertEqual(yz_av_a.domain, [])

        a = pybamm.Symbol("a", domain="bad domain")
        with self.assertRaises(pybamm.DomainError):
            pybamm.z_average(a)
        with self.assertRaises(pybamm.DomainError):
            pybamm.yz_average(a)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
