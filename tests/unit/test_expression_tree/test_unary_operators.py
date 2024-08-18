#
# Tests for the Unary Operator classes
#
import pytest

import numpy as np
from scipy.sparse import diags
import sympy
from sympy.vector.operators import Divergence as sympy_Divergence
from sympy.vector.operators import Gradient as sympy_Gradient
from tests import assert_domain_equal

import pybamm


class TestUnaryOperators:
    def test_unary_operator(self):
        a = pybamm.Symbol("a", domain=["test"])
        un = pybamm.UnaryOperator("unary test", a)
        assert un.children[0].name == a.name
        assert un.domain == a.domain

        # with number
        a = pybamm.InputParameter("a")
        absval = pybamm.AbsoluteValue(-a)
        assert absval.evaluate(inputs={"a": 10}) == 10

    def test_negation(self, mocker):
        a = pybamm.Symbol("a")
        nega = pybamm.Negate(a)
        assert nega.name == "-"
        assert nega.children[0].name == a.name

        b = pybamm.Scalar(4)
        negb = pybamm.Negate(b)
        assert negb.evaluate() == -4

        # Test broadcast gets switched
        broad_a = pybamm.PrimaryBroadcast(a, "test")
        neg_broad = -broad_a
        assert neg_broad == pybamm.PrimaryBroadcast(nega, "test")

        broad_a = pybamm.FullBroadcast(a, "test", "test2")
        neg_broad = -broad_a
        assert neg_broad == pybamm.FullBroadcast(nega, "test", "test2")

        # Test recursion
        broad_a = pybamm.PrimaryBroadcast(pybamm.PrimaryBroadcast(a, "test"), "test2")
        neg_broad = -broad_a
        assert neg_broad == \
            pybamm.PrimaryBroadcast(pybamm.PrimaryBroadcast(nega, "test"), "test2")

        # Test from_json
        input_json = {
            "name": "-",
            "id": mocker.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "children": [a],
        }
        assert pybamm.Negate._from_json(input_json) == nega

    def test_absolute(self, mocker):
        a = pybamm.Symbol("a")
        absa = pybamm.AbsoluteValue(a)
        assert absa.name == "abs"
        assert absa.children[0].name == a.name

        b = pybamm.Scalar(-4)
        absb = pybamm.AbsoluteValue(b)
        assert absb.evaluate() == 4

        # Test broadcast gets switched
        broad_a = pybamm.PrimaryBroadcast(a, "test")
        abs_broad = abs(broad_a)
        assert abs_broad == pybamm.PrimaryBroadcast(absa, "test")

        broad_a = pybamm.FullBroadcast(a, "test", "test2")
        abs_broad = abs(broad_a)
        assert abs_broad == pybamm.FullBroadcast(absa, "test", "test2")

        # Test recursion
        broad_a = pybamm.PrimaryBroadcast(pybamm.PrimaryBroadcast(a, "test"), "test2")
        abs_broad = abs(broad_a)
        assert abs_broad == \
            pybamm.PrimaryBroadcast(pybamm.PrimaryBroadcast(absa, "test"), "test2")

        # Test from_json
        input_json = {
            "name": "abs",
            "id": mocker.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "children": [a],
        }
        assert pybamm.AbsoluteValue._from_json(input_json) == absa

    def test_smooth_absolute_value(self):
        a = pybamm.StateVector(slice(0, 1))
        expr = pybamm.smooth_absolute_value(a, 10)
        assert expr.evaluate(y=np.array([1]))[0, 0] == pytest.approx(1)
        assert expr.evaluate(y=np.array([0])) == 0
        assert expr.evaluate(y=np.array([-1]))[0, 0] == pytest.approx(1)
        assert str(expr) == \
            "y[0:1] * (exp(10.0 * y[0:1]) - exp(-10.0 * y[0:1])) " \
            "/ (exp(10.0 * y[0:1]) + exp(-10.0 * y[0:1]))"

    def test_sign(self):
        b = pybamm.Scalar(-4)
        signb = pybamm.sign(b)
        assert signb.evaluate() == -1

        A = diags(np.linspace(-1, 1, 5))
        b = pybamm.Matrix(A)
        signb = pybamm.sign(b)
        np.testing.assert_array_equal(
            np.diag(signb.evaluate().toarray()), [-1, -1, 0, 1, 1]
        )

        broad = pybamm.PrimaryBroadcast(-4, "test domain")
        assert pybamm.sign(broad) == pybamm.PrimaryBroadcast(-1, "test domain")

        conc = pybamm.Concatenation(broad, pybamm.PrimaryBroadcast(2, "another domain"))
        assert pybamm.sign(conc) == \
            pybamm.Concatenation(
                pybamm.PrimaryBroadcast(-1, "test domain"),
                pybamm.PrimaryBroadcast(1, "another domain"),
            )

        # Test from_json
        with pytest.raises(NotImplementedError):
            # signs are always scalar/array types in a discretised model
            pybamm.Sign._from_json({})

    def test_floor(self, mocker):
        a = pybamm.Symbol("a")
        floora = pybamm.Floor(a)
        assert floora.name == "floor"
        assert floora.children[0].name == a.name

        b = pybamm.Scalar(3.5)
        floorb = pybamm.Floor(b)
        assert floorb.evaluate() == 3

        c = pybamm.Scalar(-3.2)
        floorc = pybamm.Floor(c)
        assert floorc.evaluate() == -4

        # Test from_json
        input_json = {
            "name": "floor",
            "id": mocker.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "children": [a],
        }
        assert pybamm.Floor._from_json(input_json) == floora

    def test_ceiling(self, mocker):
        a = pybamm.Symbol("a")
        ceila = pybamm.Ceiling(a)
        assert ceila.name == "ceil"
        assert ceila.children[0].name == a.name

        b = pybamm.Scalar(3.5)
        ceilb = pybamm.Ceiling(b)
        assert ceilb.evaluate() == 4

        c = pybamm.Scalar(-3.2)
        ceilc = pybamm.Ceiling(c)
        assert ceilc.evaluate() == -3

        # Test from_json
        input_json = {
            "name": "ceil",
            "id": mocker.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "children": [a],
        }
        assert pybamm.Ceiling._from_json(input_json) == ceila

    def test_gradient(self):
        # gradient of scalar symbol should fail
        a = pybamm.Symbol("a")
        with pytest.raises(
            pybamm.DomainError, match="Cannot take gradient of 'a' since its domain is empty"
        ):
            pybamm.Gradient(a)

        # gradient of variable evaluating on edges should fail
        a = pybamm.PrimaryBroadcastToEdges(pybamm.Scalar(1), "test")
        with pytest.raises(TypeError, match="evaluates on edges"):
            pybamm.Gradient(a)

        # gradient of broadcast should return broadcasted zero
        a = pybamm.PrimaryBroadcast(pybamm.Variable("a"), "test domain")
        grad = pybamm.grad(a)
        assert grad == pybamm.PrimaryBroadcastToEdges(0, "test domain")

        # gradient of a secondary broadcast moves the secondary out of the gradient
        a = pybamm.Symbol("a", domain="test domain")
        a_broad = pybamm.SecondaryBroadcast(a, "another domain")
        grad = pybamm.grad(a_broad)
        assert grad == pybamm.SecondaryBroadcast(pybamm.grad(a), "another domain")

        # otherwise gradient should work
        a = pybamm.Symbol("a", domain="test domain")
        grad = pybamm.Gradient(a)
        assert grad.children[0].name == a.name
        assert grad.domain == a.domain

    def test_div(self):
        # divergence of scalar symbol should fail
        a = pybamm.Symbol("a")
        with pytest.raises(
            pybamm.DomainError,
            match="Cannot take divergence of 'a' since its domain is empty",
        ):
            pybamm.Divergence(a)

        # divergence of variable evaluating on edges should fail
        a = pybamm.PrimaryBroadcast(pybamm.Scalar(1), "test")
        with pytest.raises(TypeError, match="evaluate on edges"):
            pybamm.Divergence(a)

        # divergence of broadcast should return broadcasted zero
        a = pybamm.PrimaryBroadcastToEdges(pybamm.Variable("a"), "test domain")
        div = pybamm.div(a)
        assert div == pybamm.PrimaryBroadcast(0, "test domain")
        a = pybamm.PrimaryBroadcastToEdges(
            pybamm.Variable("a", "some domain"), "test domain"
        )
        div = pybamm.div(a)
        assert div == \
            pybamm.PrimaryBroadcast(
                pybamm.PrimaryBroadcast(0, "some domain"), "test domain" \
            )

        # otherwise divergence should work
        a = pybamm.Symbol("a", domain="test domain")
        div = pybamm.Divergence(pybamm.Gradient(a))
        assert div.domain == a.domain

        # check div commutes with negation
        a = pybamm.Symbol("a", domain="test domain")
        div = pybamm.div(-pybamm.Gradient(a))
        assert div == (-pybamm.Divergence(pybamm.Gradient(a)))

        div = pybamm.div(-a * pybamm.Gradient(a))
        assert div == (-pybamm.Divergence(a * pybamm.Gradient(a)))

        # div = pybamm.div(a * -pybamm.Gradient(a))
        # self.assertEqual(div, (-pybamm.Divergence(a * pybamm.Gradient(a))))

    def test_integral(self):
        # space integral
        a = pybamm.Symbol("a", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", ["negative electrode"])
        inta = pybamm.Integral(a, x)
        assert inta.name == "integral dx ['negative electrode']"
        assert inta.children[0].name == a.name
        assert inta.integration_variable[0] == x
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
        assert inta.name == "integral dy dz ['current collector']"
        assert inta.children[0].name == b.name
        assert inta.integration_variable[0] == y
        assert inta.integration_variable[1] == z
        assert inta.domain == []

        # Indefinite
        inta = pybamm.IndefiniteIntegral(a, x)
        assert inta.name == "a integrated w.r.t x on ['negative electrode']"
        assert inta.children[0].name == a.name
        assert inta.integration_variable[0] == x
        assert inta.domain == ["negative electrode"]
        inta_sec = pybamm.IndefiniteIntegral(a_sec, x)
        assert_domain_equal(
            inta_sec.domains,
            {"primary": ["negative electrode"], "secondary": ["current collector"]},
        )
        # backward indefinite integral
        inta = pybamm.BackwardIndefiniteIntegral(a, x)
        assert inta.name == "a integrated backward w.r.t x on ['negative electrode']"

        # expected errors
        a = pybamm.Symbol("a", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", ["separator"])
        y = pybamm.Variable("y")
        z = pybamm.SpatialVariable("z", ["negative electrode"])
        with pytest.raises(pybamm.DomainError):
            pybamm.Integral(a, x)
        with pytest.raises(TypeError, match="integration_variable must be"):
            pybamm.Integral(a, y)
        with pytest.raises(
            NotImplementedError,
            match="Indefinite integral only implemented w.r.t. one variable",
        ):
            pybamm.IndefiniteIntegral(a, [x, y])

    def test_index(self):
        vec = pybamm.StateVector(slice(0, 5))
        y_test = np.array([1, 2, 3, 4, 5])
        # with integer
        ind = pybamm.Index(vec, 3)
        assert isinstance(ind, pybamm.Index)
        assert ind.slice == slice(3, 4)
        assert ind.evaluate(y=y_test) == 4
        # with -1
        ind = pybamm.Index(vec, -1)
        assert isinstance(ind, pybamm.Index)
        assert ind.slice == slice(-1, None)
        assert ind.evaluate(y=y_test) == 5
        assert ind.name == "Index[-1]"
        # with slice
        ind = pybamm.Index(vec, slice(1, 3))
        assert isinstance(ind, pybamm.Index)
        assert ind.slice == slice(1, 3)
        np.testing.assert_array_equal(ind.evaluate(y=y_test), np.array([[2], [3]]))
        # with only stop slice
        ind = pybamm.Index(vec, slice(3))
        assert isinstance(ind, pybamm.Index)
        assert ind.slice == slice(3)
        np.testing.assert_array_equal(ind.evaluate(y=y_test), np.array([[1], [2], [3]]))

        # errors
        with pytest.raises(TypeError, match="index must be integer or slice"):
            pybamm.Index(vec, 0.0)
        debug_mode = pybamm.settings.debug_mode
        pybamm.settings.debug_mode = True
        with pytest.raises(ValueError, match="slice size exceeds child size"):
            pybamm.Index(vec, 5)
        pybamm.settings.debug_mode = debug_mode

    def test_evaluate_at(self):
        a = pybamm.Symbol("a", domain=["negative electrode"])
        f = pybamm.EvaluateAt(a, 1)
        assert f.position == 1

    def test_upwind_downwind(self):
        # upwind of scalar symbol should fail
        a = pybamm.Symbol("a")
        with pytest.raises(
            pybamm.DomainError, match="Cannot upwind 'a' since its domain is empty"
        ):
            pybamm.Upwind(a)

        # upwind of variable evaluating on edges should fail
        a = pybamm.PrimaryBroadcastToEdges(pybamm.Scalar(1), "test")
        with pytest.raises(TypeError, match="evaluate on nodes"):
            pybamm.Upwind(a)

        # otherwise upwind should work
        a = pybamm.Symbol("a", domain="test domain")
        upwind = pybamm.upwind(a)
        assert isinstance(upwind, pybamm.Upwind)
        assert upwind.children[0].name == a.name
        assert upwind.domain == a.domain

        # also test downwind
        a = pybamm.Symbol("a", domain="test domain")
        downwind = pybamm.downwind(a)
        assert isinstance(downwind, pybamm.Downwind)
        assert downwind.children[0].name == a.name
        assert downwind.domain == a.domain

    def test_diff(self):
        a = pybamm.StateVector(slice(0, 1))
        y = np.array([5])

        # negation
        assert (-a).diff(a).evaluate(y=y) == -1
        assert (-a).diff(-a).evaluate() == 1

        # absolute value
        assert (a**3).diff(a).evaluate(y=y) == 3 * 5**2
        assert (abs(a**3)).diff(a).evaluate(y=y) == 3 * 5**2
        assert (a**3).diff(a).evaluate(y=-y) == 3 * 5**2
        assert (abs(a**3)).diff(a).evaluate(y=-y) == -3 * 5**2

        # sign
        assert (pybamm.sign(a)).diff(a).evaluate(y=y) == 0

        # floor
        assert (pybamm.Floor(a)).diff(a).evaluate(y=y) == 0

        # ceil
        assert (pybamm.Ceiling(a)).diff(a).evaluate(y=y) == 0

        # spatial operator (not implemented)
        spatial_a = pybamm.SpatialOperator("name", a)
        with pytest.raises(NotImplementedError):
            spatial_a.diff(a)

    def test_printing(self):
        a = pybamm.Symbol("a", domain="test")
        assert str(-a) == "-a"
        grad = pybamm.Gradient(a)
        assert grad.name == "grad"
        assert str(grad) == "grad(a)"

    def test_eq(self):
        a = pybamm.Scalar(4)
        un1 = pybamm.UnaryOperator("test", a)
        un2 = pybamm.UnaryOperator("test", a)
        un3 = pybamm.UnaryOperator("new test", a)
        assert un1 == un2
        assert un1 != un3
        a = pybamm.Scalar(4)
        un4 = pybamm.UnaryOperator("test", a)
        assert un1 == un4
        d = pybamm.Scalar(42)
        un5 = pybamm.UnaryOperator("test", d)
        assert un1 != un5

    def test_delta_function(self):
        a = pybamm.Symbol("a")
        delta_a = pybamm.DeltaFunction(a, "right", "some domain")
        assert delta_a.side == "right"
        assert delta_a.child == a
        assert delta_a.domain == ["some domain"]
        assert not delta_a.evaluates_on_edges("primary")

        a = pybamm.Symbol("a", domain="some domain")
        delta_a = pybamm.DeltaFunction(a, "left", "another domain")
        assert delta_a.side == "left"
        assert_domain_equal(
            delta_a.domains,
            {"primary": ["another domain"], "secondary": ["some domain"]},
        )

        with pytest.raises(
            pybamm.DomainError, match="Delta function domain cannot be None"
        ):
            delta_a = pybamm.DeltaFunction(a, "right", None)

    def test_boundary_operators(self):
        a = pybamm.Symbol("a", domain="some domain")
        boundary_a = pybamm.BoundaryOperator("boundary", a, "right")
        assert boundary_a.side == "right"
        assert boundary_a.child == a

    def test_evaluates_on_edges(self):
        a = pybamm.StateVector(slice(0, 10), domain="test")
        assert not pybamm.Index(a, slice(1)).evaluates_on_edges("primary")
        assert not pybamm.Laplacian(a).evaluates_on_edges("primary")
        assert not pybamm.GradientSquared(a).evaluates_on_edges("primary")
        assert not pybamm.BoundaryIntegral(a).evaluates_on_edges("primary")
        assert pybamm.Upwind(a).evaluates_on_edges("primary")
        assert pybamm.Downwind(a).evaluates_on_edges("primary")

    def test_boundary_value(self):
        a = pybamm.Scalar(1)
        boundary_a = pybamm.boundary_value(a, "right")
        assert boundary_a == a

        boundary_broad_a = pybamm.boundary_value(
            pybamm.PrimaryBroadcast(a, ["negative electrode"]), "left"
        )
        assert boundary_broad_a.evaluate() == np.array([1])

        a = pybamm.Symbol("a", domain=["separator"])
        boundary_a = pybamm.boundary_value(a, "right")
        assert isinstance(boundary_a, pybamm.BoundaryValue)
        assert boundary_a.side == "right"
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
        assert boundary_a_quat.domain == ["current collector"]
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
        with pytest.raises(pybamm.ModelError, match="Can only take boundary"):
            pybamm.boundary_value(var, "negative tab")
            pybamm.boundary_value(var, "positive tab")

        # boundary value of symbol that evaluates on edges raises error
        symbol_on_edges = pybamm.PrimaryBroadcastToEdges(1, "domain")
        with pytest.raises(
            ValueError,
            match="Can't take the boundary value of a symbol that evaluates on edges",
        ):
            pybamm.boundary_value(symbol_on_edges, "right")

    def test_boundary_gradient(self):
        var = pybamm.Variable("var", domain=["negative electrode"])
        grad = pybamm.boundary_gradient(var, "right")
        assert isinstance(grad, pybamm.BoundaryGradient)

        zero = pybamm.PrimaryBroadcast(0, ["negative electrode"])
        grad = pybamm.boundary_gradient(zero, "right")
        assert grad == 0

    def test_unary_simplifications(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)
        d = pybamm.Scalar(-1)

        # negate
        assert isinstance((-a), pybamm.Scalar)
        assert (-a).evaluate() == 0
        assert isinstance((-b), pybamm.Scalar)
        assert (-b).evaluate() == -1

        # absolute value
        assert isinstance((abs(a)), pybamm.Scalar)
        assert (abs(a)).evaluate() == 0
        assert isinstance((abs(d)), pybamm.Scalar)
        assert (abs(d)).evaluate() == 1

    def test_not_constant(self):
        a = pybamm.NotConstant(pybamm.Scalar(1))
        assert a.name == "not_constant"
        assert a.domain == []
        assert a.evaluate() == 1
        assert a.jac(pybamm.StateVector(slice(0, 1))).evaluate() == 0
        assert not a.is_constant()
        assert not (2 * a).is_constant()

    def test_to_equation(self):
        a = pybamm.Symbol("a", domain="negative particle")
        b = pybamm.Symbol("b", domain="current collector")
        c = pybamm.Symbol("c", domain="test")
        d = pybamm.Symbol("d", domain=["negative electrode"])
        one = pybamm.Symbol("1", domain="negative particle")

        # Test print_name
        pybamm.Floor.print_name = "test"
        assert pybamm.Floor(-2.5).to_equation() == sympy.Symbol("test")

        # Test Negate
        value = 4
        assert pybamm.Negate(value).to_equation() == -value

        # Test AbsoluteValue
        assert pybamm.AbsoluteValue(-value).to_equation() == value

        # Test Gradient
        assert pybamm.Gradient(a).to_equation() == sympy_Gradient("a")

        # Test Divergence
        assert pybamm.Divergence(pybamm.Gradient(a)).to_equation() == \
            sympy_Divergence(sympy_Gradient("a"))

        # Test BoundaryValue
        assert pybamm.BoundaryValue(one, "right").to_equation() == sympy.Symbol("1")
        assert pybamm.BoundaryValue(a, "right").to_equation() == sympy.Symbol("a^{surf}")
        assert pybamm.BoundaryValue(b, "positive tab").to_equation() == sympy.Symbol(str(b))
        assert pybamm.BoundaryValue(c, "left").to_equation() == \
            sympy.Symbol(r"c^{\mathtt{\text{left}}}")

        # Test Integral
        xn = pybamm.SpatialVariable("xn", ["negative electrode"])
        assert pybamm.Integral(d, xn).to_equation() == \
            sympy.Integral("d", sympy.Symbol("xn"))

    def test_explicit_time_integral(self):
        expr = pybamm.ExplicitTimeIntegral(pybamm.Parameter("param"), pybamm.Scalar(1))
        assert expr.child == pybamm.Parameter("param")
        assert expr.initial_condition == pybamm.Scalar(1)
        assert expr.name == "explicit time integral"
        assert expr.create_copy() == expr
        assert not expr.is_constant()

    def test_to_from_json(self, mocker):
        # UnaryOperator
        a = pybamm.Symbol("a", domain=["test"])
        un = pybamm.UnaryOperator("unary test", a)

        un_json = {
            "name": "unary test",
            "id": mocker.ANY,
            "domains": {
                "primary": ["test"],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
        }

        assert un.to_json() == un_json

        un_json["children"] = [a]
        assert pybamm.UnaryOperator._from_json(un_json) == un

        # Index
        vec = pybamm.StateVector(slice(0, 5))
        ind = pybamm.Index(vec, 3)

        ind_json = {
            "name": "Index[3]",
            "id": mocker.ANY,
            "index": {"start": 3, "stop": 4, "step": None},
            "check_size": False,
        }

        assert ind.to_json() == ind_json

        ind_json["children"] = [vec]
        assert pybamm.Index._from_json(ind_json) == ind

        # SpatialOperator
        spatial_vec = pybamm.SpatialOperator("name", vec)
        with pytest.raises(NotImplementedError):
            spatial_vec.to_json()

        with pytest.raises(NotImplementedError):
            pybamm.SpatialOperator._from_json({})

        # ExplicitTimeIntegral
        expr = pybamm.ExplicitTimeIntegral(pybamm.Parameter("param"), pybamm.Scalar(1))

        expr_json = {"name": "explicit time integral", "id": mocker.ANY}

        assert expr.to_json() == expr_json

        expr_json["children"] = [pybamm.Parameter("param")]
        expr_json["initial_condition"] = [pybamm.Scalar(1)]
        assert pybamm.ExplicitTimeIntegral._from_json(expr_json) == expr

