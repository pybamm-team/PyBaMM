#
# Tests for the Unary Operator classes
#
import unittest

import unittest.mock as mock

import numpy as np
from scipy.sparse import diags
import sympy
from sympy.vector.operators import Divergence as sympy_Divergence
from sympy.vector.operators import Gradient as sympy_Gradient
from tests import assert_domain_equal

import pybamm


class TestUnaryOperators(unittest.TestCase):
    def test_unary_operator(self):
        a = pybamm.Symbol("a", domain=["test"])
        un = pybamm.UnaryOperator("unary test", a)
        self.assertEqual(un.children[0].name, a.name)
        self.assertEqual(un.domain, a.domain)

        # with number
        a = pybamm.InputParameter("a")
        absval = pybamm.AbsoluteValue(-a)
        self.assertEqual(absval.evaluate(inputs={"a": 10}), 10)

    def test_negation(self):
        a = pybamm.Symbol("a")
        nega = pybamm.Negate(a)
        self.assertEqual(nega.name, "-")
        self.assertEqual(nega.children[0].name, a.name)

        b = pybamm.Scalar(4)
        negb = pybamm.Negate(b)
        self.assertEqual(negb.evaluate(), -4)

        # Test broadcast gets switched
        broad_a = pybamm.PrimaryBroadcast(a, "test")
        neg_broad = -broad_a
        self.assertEqual(neg_broad, pybamm.PrimaryBroadcast(nega, "test"))

        broad_a = pybamm.FullBroadcast(a, "test", "test2")
        neg_broad = -broad_a
        self.assertEqual(neg_broad, pybamm.FullBroadcast(nega, "test", "test2"))

        # Test recursion
        broad_a = pybamm.PrimaryBroadcast(pybamm.PrimaryBroadcast(a, "test"), "test2")
        neg_broad = -broad_a
        self.assertEqual(
            neg_broad,
            pybamm.PrimaryBroadcast(pybamm.PrimaryBroadcast(nega, "test"), "test2"),
        )

        # Test from_json
        input_json = {
            "name": "-",
            "id": mock.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "children": [a],
        }
        self.assertEqual(pybamm.Negate._from_json(input_json), nega)

    def test_absolute(self):
        a = pybamm.Symbol("a")
        absa = pybamm.AbsoluteValue(a)
        self.assertEqual(absa.name, "abs")
        self.assertEqual(absa.children[0].name, a.name)

        b = pybamm.Scalar(-4)
        absb = pybamm.AbsoluteValue(b)
        self.assertEqual(absb.evaluate(), 4)

        # Test broadcast gets switched
        broad_a = pybamm.PrimaryBroadcast(a, "test")
        abs_broad = abs(broad_a)
        self.assertEqual(abs_broad, pybamm.PrimaryBroadcast(absa, "test"))

        broad_a = pybamm.FullBroadcast(a, "test", "test2")
        abs_broad = abs(broad_a)
        self.assertEqual(abs_broad, pybamm.FullBroadcast(absa, "test", "test2"))

        # Test recursion
        broad_a = pybamm.PrimaryBroadcast(pybamm.PrimaryBroadcast(a, "test"), "test2")
        abs_broad = abs(broad_a)
        self.assertEqual(
            abs_broad,
            pybamm.PrimaryBroadcast(pybamm.PrimaryBroadcast(absa, "test"), "test2"),
        )

        # Test from_json
        input_json = {
            "name": "abs",
            "id": mock.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "children": [a],
        }
        self.assertEqual(pybamm.AbsoluteValue._from_json(input_json), absa)

    def test_smooth_absolute_value(self):
        a = pybamm.StateVector(slice(0, 1))
        expr = pybamm.smooth_absolute_value(a, 10)
        self.assertAlmostEqual(expr.evaluate(y=np.array([1]))[0, 0], 1)
        self.assertEqual(expr.evaluate(y=np.array([0])), 0)
        self.assertAlmostEqual(expr.evaluate(y=np.array([-1]))[0, 0], 1)
        self.assertEqual(
            str(expr),
            "y[0:1] * (exp(10.0 * y[0:1]) - exp(-10.0 * y[0:1])) "
            "/ (exp(10.0 * y[0:1]) + exp(-10.0 * y[0:1]))",
        )

    def test_sign(self):
        b = pybamm.Scalar(-4)
        signb = pybamm.sign(b)
        self.assertEqual(signb.evaluate(), -1)

        A = diags(np.linspace(-1, 1, 5))
        b = pybamm.Matrix(A)
        signb = pybamm.sign(b)
        np.testing.assert_array_equal(
            np.diag(signb.evaluate().toarray()), [-1, -1, 0, 1, 1]
        )

        broad = pybamm.PrimaryBroadcast(-4, "test domain")
        self.assertEqual(pybamm.sign(broad), pybamm.PrimaryBroadcast(-1, "test domain"))

        conc = pybamm.Concatenation(broad, pybamm.PrimaryBroadcast(2, "another domain"))
        self.assertEqual(
            pybamm.sign(conc),
            pybamm.Concatenation(
                pybamm.PrimaryBroadcast(-1, "test domain"),
                pybamm.PrimaryBroadcast(1, "another domain"),
            ),
        )

        # Test from_json
        with self.assertRaises(NotImplementedError):
            # signs are always scalar/array types in a discretised model
            pybamm.Sign._from_json({})

    def test_floor(self):
        a = pybamm.Symbol("a")
        floora = pybamm.Floor(a)
        self.assertEqual(floora.name, "floor")
        self.assertEqual(floora.children[0].name, a.name)

        b = pybamm.Scalar(3.5)
        floorb = pybamm.Floor(b)
        self.assertEqual(floorb.evaluate(), 3)

        c = pybamm.Scalar(-3.2)
        floorc = pybamm.Floor(c)
        self.assertEqual(floorc.evaluate(), -4)

        # Test from_json
        input_json = {
            "name": "floor",
            "id": mock.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "children": [a],
        }
        self.assertEqual(pybamm.Floor._from_json(input_json), floora)

    def test_ceiling(self):
        a = pybamm.Symbol("a")
        ceila = pybamm.Ceiling(a)
        self.assertEqual(ceila.name, "ceil")
        self.assertEqual(ceila.children[0].name, a.name)

        b = pybamm.Scalar(3.5)
        ceilb = pybamm.Ceiling(b)
        self.assertEqual(ceilb.evaluate(), 4)

        c = pybamm.Scalar(-3.2)
        ceilc = pybamm.Ceiling(c)
        self.assertEqual(ceilc.evaluate(), -3)

        # Test from_json
        input_json = {
            "name": "ceil",
            "id": mock.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "children": [a],
        }
        self.assertEqual(pybamm.Ceiling._from_json(input_json), ceila)

    def test_gradient(self):
        # gradient of scalar symbol should fail
        a = pybamm.Symbol("a")
        with self.assertRaisesRegex(
            pybamm.DomainError, "Cannot take gradient of 'a' since its domain is empty"
        ):
            pybamm.Gradient(a)

        # gradient of variable evaluating on edges should fail
        a = pybamm.PrimaryBroadcastToEdges(pybamm.Scalar(1), "test")
        with self.assertRaisesRegex(TypeError, "evaluates on edges"):
            pybamm.Gradient(a)

        # gradient of broadcast should return broadcasted zero
        a = pybamm.PrimaryBroadcast(pybamm.Variable("a"), "test domain")
        grad = pybamm.grad(a)
        self.assertEqual(grad, pybamm.PrimaryBroadcastToEdges(0, "test domain"))

        # gradient of a secondary broadcast moves the secondary out of the gradient
        a = pybamm.Symbol("a", domain="test domain")
        a_broad = pybamm.SecondaryBroadcast(a, "another domain")
        grad = pybamm.grad(a_broad)
        self.assertEqual(
            grad, pybamm.SecondaryBroadcast(pybamm.grad(a), "another domain")
        )

        # otherwise gradient should work
        a = pybamm.Symbol("a", domain="test domain")
        grad = pybamm.Gradient(a)
        self.assertEqual(grad.children[0].name, a.name)
        self.assertEqual(grad.domain, a.domain)

    def test_div(self):
        # divergence of scalar symbol should fail
        a = pybamm.Symbol("a")
        with self.assertRaisesRegex(
            pybamm.DomainError,
            "Cannot take divergence of 'a' since its domain is empty",
        ):
            pybamm.Divergence(a)

        # divergence of variable evaluating on edges should fail
        a = pybamm.PrimaryBroadcast(pybamm.Scalar(1), "test")
        with self.assertRaisesRegex(TypeError, "evaluate on edges"):
            pybamm.Divergence(a)

        # divergence of broadcast should return broadcasted zero
        a = pybamm.PrimaryBroadcastToEdges(pybamm.Variable("a"), "test domain")
        div = pybamm.div(a)
        self.assertEqual(div, pybamm.PrimaryBroadcast(0, "test domain"))
        a = pybamm.PrimaryBroadcastToEdges(
            pybamm.Variable("a", "some domain"), "test domain"
        )
        div = pybamm.div(a)
        self.assertEqual(
            div,
            pybamm.PrimaryBroadcast(
                pybamm.PrimaryBroadcast(0, "some domain"), "test domain"
            ),
        )

        # otherwise divergence should work
        a = pybamm.Symbol("a", domain="test domain")
        div = pybamm.Divergence(pybamm.Gradient(a))
        self.assertEqual(div.domain, a.domain)

        # check div commutes with negation
        a = pybamm.Symbol("a", domain="test domain")
        div = pybamm.div(-pybamm.Gradient(a))
        self.assertEqual(div, (-pybamm.Divergence(pybamm.Gradient(a))))

        div = pybamm.div(-a * pybamm.Gradient(a))
        self.assertEqual(div, (-pybamm.Divergence(a * pybamm.Gradient(a))))

        # div = pybamm.div(a * -pybamm.Gradient(a))
        # self.assertEqual(div, (-pybamm.Divergence(a * pybamm.Gradient(a))))

    def test_integral(self):
        # space integral
        a = pybamm.Symbol("a", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", ["negative electrode"])
        inta = pybamm.Integral(a, x)
        self.assertEqual(inta.name, "integral dx ['negative electrode']")
        self.assertEqual(inta.children[0].name, a.name)
        self.assertEqual(inta.integration_variable[0], x)
        assert_domain_equal(inta.domains, {})
        # space integral with secondary domain
        a_sec = pybamm.Symbol(
            "a",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        x = pybamm.SpatialVariable("x", ["negative electrode"])
        inta_sec = pybamm.Integral(a_sec, x)
        assert_domain_equal(inta_sec.domains, {"primary": ["current collector"]})
        # space integral with tertiary domain
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
        assert_domain_equal(
            inta_tert.domains,
            {"primary": ["current collector"], "secondary": ["some extra domain"]},
        )
        # space integral with quaternary domain
        a_quat = pybamm.Symbol(
            "a",
            domain=["negative electrode"],
            auxiliary_domains={
                "secondary": "current collector",
                "tertiary": "some extra domain",
                "quaternary": "another extra domain",
            },
        )
        inta_quat = pybamm.Integral(a_quat, x)
        assert_domain_equal(
            inta_quat.domains,
            {
                "primary": ["current collector"],
                "secondary": ["some extra domain"],
                "tertiary": ["another extra domain"],
            },
        )

        # space integral *in* secondary domain
        y = pybamm.SpatialVariable("y", ["current collector"])
        # without a tertiary domain
        inta_sec_y = pybamm.Integral(a_sec, y)
        assert_domain_equal(inta_sec_y.domains, {"primary": ["negative electrode"]})
        # with a tertiary domain
        inta_tert_y = pybamm.Integral(a_tert, y)
        assert_domain_equal(
            inta_tert_y.domains,
            {"primary": ["negative electrode"], "secondary": ["some extra domain"]},
        )
        # with a quaternary domain
        inta_quat_y = pybamm.Integral(a_quat, y)
        assert_domain_equal(
            inta_quat_y.domains,
            {
                "primary": ["negative electrode"],
                "secondary": ["some extra domain"],
                "tertiary": ["another extra domain"],
            },
        )

        # space integral *in* tertiary domain
        z = pybamm.SpatialVariable("z", ["some extra domain"])
        inta_tert_z = pybamm.Integral(a_tert, z)
        assert_domain_equal(
            inta_tert_z.domains,
            {"primary": ["negative electrode"], "secondary": ["current collector"]},
        )
        # with a quaternary domain
        inta_quat_z = pybamm.Integral(a_quat, z)
        assert_domain_equal(
            inta_quat_z.domains,
            {
                "primary": ["negative electrode"],
                "secondary": ["current collector"],
                "tertiary": ["another extra domain"],
            },
        )

        # space integral *in* quaternary domain
        Z = pybamm.SpatialVariable("Z", ["another extra domain"])
        inta_quat_Z = pybamm.Integral(a_quat, Z)
        assert_domain_equal(
            inta_quat_Z.domains,
            {
                "primary": ["negative electrode"],
                "secondary": ["current collector"],
                "tertiary": ["some extra domain"],
            },
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
        assert_domain_equal(
            inta_sec.domains,
            {"primary": ["negative electrode"], "secondary": ["current collector"]},
        )
        # backward indefinite integral
        inta = pybamm.BackwardIndefiniteIntegral(a, x)
        self.assertEqual(
            inta.name, "a integrated backward w.r.t x on ['negative electrode']"
        )

        # expected errors
        a = pybamm.Symbol("a", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", ["separator"])
        y = pybamm.Variable("y")
        z = pybamm.SpatialVariable("z", ["negative electrode"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.Integral(a, x)
        with self.assertRaisesRegex(TypeError, "integration_variable must be"):
            pybamm.Integral(a, y)
        with self.assertRaisesRegex(
            NotImplementedError,
            "Indefinite integral only implemented w.r.t. one variable",
        ):
            pybamm.IndefiniteIntegral(a, [x, y])

    def test_index(self):
        vec = pybamm.StateVector(slice(0, 5))
        y_test = np.array([1, 2, 3, 4, 5])
        # with integer
        ind = pybamm.Index(vec, 3)
        self.assertIsInstance(ind, pybamm.Index)
        self.assertEqual(ind.slice, slice(3, 4))
        self.assertEqual(ind.evaluate(y=y_test), 4)
        # with -1
        ind = pybamm.Index(vec, -1)
        self.assertIsInstance(ind, pybamm.Index)
        self.assertEqual(ind.slice, slice(-1, None))
        self.assertEqual(ind.evaluate(y=y_test), 5)
        self.assertEqual(ind.name, "Index[-1]")
        # with slice
        ind = pybamm.Index(vec, slice(1, 3))
        self.assertIsInstance(ind, pybamm.Index)
        self.assertEqual(ind.slice, slice(1, 3))
        np.testing.assert_array_equal(ind.evaluate(y=y_test), np.array([[2], [3]]))
        # with only stop slice
        ind = pybamm.Index(vec, slice(3))
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

    def test_evaluate_at(self):
        a = pybamm.Symbol("a", domain=["negative electrode"])
        f = pybamm.EvaluateAt(a, 1)
        self.assertEqual(f.position, 1)

    def test_upwind_downwind(self):
        # upwind of scalar symbol should fail
        a = pybamm.Symbol("a")
        with self.assertRaisesRegex(
            pybamm.DomainError, "Cannot upwind 'a' since its domain is empty"
        ):
            pybamm.Upwind(a)

        # upwind of variable evaluating on edges should fail
        a = pybamm.PrimaryBroadcastToEdges(pybamm.Scalar(1), "test")
        with self.assertRaisesRegex(TypeError, "evaluate on nodes"):
            pybamm.Upwind(a)

        # otherwise upwind should work
        a = pybamm.Symbol("a", domain="test domain")
        upwind = pybamm.upwind(a)
        self.assertIsInstance(upwind, pybamm.Upwind)
        self.assertEqual(upwind.children[0].name, a.name)
        self.assertEqual(upwind.domain, a.domain)

        # also test downwind
        a = pybamm.Symbol("a", domain="test domain")
        downwind = pybamm.downwind(a)
        self.assertIsInstance(downwind, pybamm.Downwind)
        self.assertEqual(downwind.children[0].name, a.name)
        self.assertEqual(downwind.domain, a.domain)

    def test_diff(self):
        a = pybamm.StateVector(slice(0, 1))
        y = np.array([5])

        # negation
        self.assertEqual((-a).diff(a).evaluate(y=y), -1)
        self.assertEqual((-a).diff(-a).evaluate(), 1)

        # absolute value
        self.assertEqual((a**3).diff(a).evaluate(y=y), 3 * 5**2)
        self.assertEqual((abs(a**3)).diff(a).evaluate(y=y), 3 * 5**2)
        self.assertEqual((a**3).diff(a).evaluate(y=-y), 3 * 5**2)
        self.assertEqual((abs(a**3)).diff(a).evaluate(y=-y), -3 * 5**2)

        # sign
        self.assertEqual((pybamm.sign(a)).diff(a).evaluate(y=y), 0)

        # floor
        self.assertEqual((pybamm.Floor(a)).diff(a).evaluate(y=y), 0)

        # ceil
        self.assertEqual((pybamm.Ceiling(a)).diff(a).evaluate(y=y), 0)

        # spatial operator (not implemented)
        spatial_a = pybamm.SpatialOperator("name", a)
        with self.assertRaises(NotImplementedError):
            spatial_a.diff(a)

    def test_printing(self):
        a = pybamm.Symbol("a", domain="test")
        self.assertEqual(str(-a), "-a")
        grad = pybamm.Gradient(a)
        self.assertEqual(grad.name, "grad")
        self.assertEqual(str(grad), "grad(a)")

    def test_eq(self):
        a = pybamm.Scalar(4)
        un1 = pybamm.UnaryOperator("test", a)
        un2 = pybamm.UnaryOperator("test", a)
        un3 = pybamm.UnaryOperator("new test", a)
        self.assertEqual(un1, un2)
        self.assertNotEqual(un1, un3)
        a = pybamm.Scalar(4)
        un4 = pybamm.UnaryOperator("test", a)
        self.assertEqual(un1, un4)
        d = pybamm.Scalar(42)
        un5 = pybamm.UnaryOperator("test", d)
        self.assertNotEqual(un1, un5)

    def test_delta_function(self):
        a = pybamm.Symbol("a")
        delta_a = pybamm.DeltaFunction(a, "right", "some domain")
        self.assertEqual(delta_a.side, "right")
        self.assertEqual(delta_a.child, a)
        self.assertEqual(delta_a.domain, ["some domain"])
        self.assertFalse(delta_a.evaluates_on_edges("primary"))

        a = pybamm.Symbol("a", domain="some domain")
        delta_a = pybamm.DeltaFunction(a, "left", "another domain")
        self.assertEqual(delta_a.side, "left")
        assert_domain_equal(
            delta_a.domains,
            {"primary": ["another domain"], "secondary": ["some domain"]},
        )

        with self.assertRaisesRegex(
            pybamm.DomainError, "Delta function domain cannot be None"
        ):
            delta_a = pybamm.DeltaFunction(a, "right", None)

    def test_boundary_operators(self):
        a = pybamm.Symbol("a", domain="some domain")
        boundary_a = pybamm.BoundaryOperator("boundary", a, "right")
        self.assertEqual(boundary_a.side, "right")
        self.assertEqual(boundary_a.child, a)

    def test_evaluates_on_edges(self):
        a = pybamm.StateVector(slice(0, 10), domain="test")
        self.assertFalse(pybamm.Index(a, slice(1)).evaluates_on_edges("primary"))
        self.assertFalse(pybamm.Laplacian(a).evaluates_on_edges("primary"))
        self.assertFalse(pybamm.GradientSquared(a).evaluates_on_edges("primary"))
        self.assertFalse(pybamm.BoundaryIntegral(a).evaluates_on_edges("primary"))
        self.assertTrue(pybamm.Upwind(a).evaluates_on_edges("primary"))
        self.assertTrue(pybamm.Downwind(a).evaluates_on_edges("primary"))

    def test_boundary_value(self):
        a = pybamm.Scalar(1)
        boundary_a = pybamm.boundary_value(a, "right")
        self.assertEqual(boundary_a, a)

        boundary_broad_a = pybamm.boundary_value(
            pybamm.PrimaryBroadcast(a, ["negative electrode"]), "left"
        )
        self.assertEqual(boundary_broad_a.evaluate(), np.array([1]))

        a = pybamm.Symbol("a", domain=["separator"])
        boundary_a = pybamm.boundary_value(a, "right")
        self.assertIsInstance(boundary_a, pybamm.BoundaryValue)
        self.assertEqual(boundary_a.side, "right")
        assert_domain_equal(boundary_a.domains, {})
        # test with secondary domain
        a_sec = pybamm.Symbol(
            "a",
            domain=["separator"],
            auxiliary_domains={"secondary": "current collector"},
        )
        boundary_a_sec = pybamm.boundary_value(a_sec, "right")
        assert_domain_equal(boundary_a_sec.domains, {"primary": ["current collector"]})
        # test with secondary domain and tertiary domain
        a_tert = pybamm.Symbol(
            "a",
            domain=["separator"],
            auxiliary_domains={"secondary": "current collector", "tertiary": "bla"},
        )
        boundary_a_tert = pybamm.boundary_value(a_tert, "right")
        assert_domain_equal(
            boundary_a_tert.domains,
            {"primary": ["current collector"], "secondary": ["bla"]},
        )
        # test with secondary, tertiary and quaternary domains
        a_quat = pybamm.Symbol(
            "a",
            domain=["separator"],
            auxiliary_domains={
                "secondary": "current collector",
                "tertiary": "bla",
                "quaternary": "another domain",
            },
        )
        boundary_a_quat = pybamm.boundary_value(a_quat, "right")
        self.assertEqual(boundary_a_quat.domain, ["current collector"])
        assert_domain_equal(
            boundary_a_quat.domains,
            {
                "primary": ["current collector"],
                "secondary": ["bla"],
                "tertiary": ["another domain"],
            },
        )

        # error if boundary value on tabs and domain is not "current collector"
        var = pybamm.Variable("var", domain=["negative electrode"])
        with self.assertRaisesRegex(pybamm.ModelError, "Can only take boundary"):
            pybamm.boundary_value(var, "negative tab")
            pybamm.boundary_value(var, "positive tab")

        # boundary value of symbol that evaluates on edges raises error
        symbol_on_edges = pybamm.PrimaryBroadcastToEdges(1, "domain")
        with self.assertRaisesRegex(
            ValueError,
            "Can't take the boundary value of a symbol that evaluates on edges",
        ):
            pybamm.boundary_value(symbol_on_edges, "right")

    def test_boundary_gradient(self):
        var = pybamm.Variable("var", domain=["negative electrode"])
        grad = pybamm.boundary_gradient(var, "right")
        self.assertIsInstance(grad, pybamm.BoundaryGradient)

        zero = pybamm.PrimaryBroadcast(0, ["negative electrode"])
        grad = pybamm.boundary_gradient(zero, "right")
        self.assertEqual(grad, 0)

    def test_unary_simplifications(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)
        d = pybamm.Scalar(-1)

        # negate
        self.assertIsInstance((-a), pybamm.Scalar)
        self.assertEqual((-a).evaluate(), 0)
        self.assertIsInstance((-b), pybamm.Scalar)
        self.assertEqual((-b).evaluate(), -1)

        # absolute value
        self.assertIsInstance((abs(a)), pybamm.Scalar)
        self.assertEqual((abs(a)).evaluate(), 0)
        self.assertIsInstance((abs(d)), pybamm.Scalar)
        self.assertEqual((abs(d)).evaluate(), 1)

    def test_not_constant(self):
        a = pybamm.NotConstant(pybamm.Scalar(1))
        self.assertEqual(a.name, "not_constant")
        self.assertEqual(a.domain, [])
        self.assertEqual(a.evaluate(), 1)
        self.assertEqual(a.jac(pybamm.StateVector(slice(0, 1))).evaluate(), 0)
        self.assertFalse(a.is_constant())
        self.assertFalse((2 * a).is_constant())

    def test_to_equation(self):
        a = pybamm.Symbol("a", domain="negative particle")
        b = pybamm.Symbol("b", domain="current collector")
        c = pybamm.Symbol("c", domain="test")
        d = pybamm.Symbol("d", domain=["negative electrode"])
        one = pybamm.Symbol("1", domain="negative particle")

        # Test print_name
        pybamm.Floor.print_name = "test"
        self.assertEqual(pybamm.Floor(-2.5).to_equation(), sympy.Symbol("test"))

        # Test Negate
        value = 4
        self.assertEqual(pybamm.Negate(value).to_equation(), -value)

        # Test AbsoluteValue
        self.assertEqual(pybamm.AbsoluteValue(-value).to_equation(), value)

        # Test Gradient
        self.assertEqual(pybamm.Gradient(a).to_equation(), sympy_Gradient("a"))

        # Test Divergence
        self.assertEqual(
            pybamm.Divergence(pybamm.Gradient(a)).to_equation(),
            sympy_Divergence(sympy_Gradient("a")),
        )

        # Test BoundaryValue
        self.assertEqual(
            pybamm.BoundaryValue(one, "right").to_equation(), sympy.Symbol("1")
        )
        self.assertEqual(
            pybamm.BoundaryValue(a, "right").to_equation(), sympy.Symbol("a^{surf}")
        )
        self.assertEqual(
            pybamm.BoundaryValue(b, "positive tab").to_equation(), sympy.Symbol(str(b))
        )
        self.assertEqual(
            pybamm.BoundaryValue(c, "left").to_equation(),
            sympy.Symbol(r"c^{\mathtt{\text{left}}}"),
        )

        # Test Integral
        xn = pybamm.SpatialVariable("xn", ["negative electrode"])
        self.assertEqual(
            pybamm.Integral(d, xn).to_equation(),
            sympy.Integral("d", sympy.Symbol("xn")),
        )

    def test_explicit_time_integral(self):
        expr = pybamm.ExplicitTimeIntegral(pybamm.Parameter("param"), pybamm.Scalar(1))
        self.assertEqual(expr.child, pybamm.Parameter("param"))
        self.assertEqual(expr.initial_condition, pybamm.Scalar(1))
        self.assertEqual(expr.name, "explicit time integral")
        self.assertEqual(expr.create_copy(), expr)
        self.assertFalse(expr.is_constant())

    def test_to_from_json(self):
        # UnaryOperator
        a = pybamm.Symbol("a", domain=["test"])
        un = pybamm.UnaryOperator("unary test", a)

        un_json = {
            "name": "unary test",
            "id": mock.ANY,
            "domains": {
                "primary": ["test"],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
        }

        self.assertEqual(un.to_json(), un_json)

        un_json["children"] = [a]
        self.assertEqual(pybamm.UnaryOperator._from_json(un_json), un)

        # Index
        vec = pybamm.StateVector(slice(0, 5))
        ind = pybamm.Index(vec, 3)

        ind_json = {
            "name": "Index[3]",
            "id": mock.ANY,
            "index": {"start": 3, "stop": 4, "step": None},
            "check_size": False,
        }

        self.assertEqual(ind.to_json(), ind_json)

        ind_json["children"] = [vec]
        self.assertEqual(pybamm.Index._from_json(ind_json), ind)

        # SpatialOperator
        spatial_vec = pybamm.SpatialOperator("name", vec)
        with self.assertRaises(NotImplementedError):
            spatial_vec.to_json()

        with self.assertRaises(NotImplementedError):
            pybamm.SpatialOperator._from_json({})

        # ExplicitTimeIntegral
        expr = pybamm.ExplicitTimeIntegral(pybamm.Parameter("param"), pybamm.Scalar(1))

        expr_json = {"name": "explicit time integral", "id": mock.ANY}

        self.assertEqual(expr.to_json(), expr_json)

        expr_json["children"] = [pybamm.Parameter("param")]
        expr_json["initial_condition"] = [pybamm.Scalar(1)]
        self.assertEqual(pybamm.ExplicitTimeIntegral._from_json(expr_json), expr)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
