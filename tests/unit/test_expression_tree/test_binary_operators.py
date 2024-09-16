#
# Tests for the Binary Operator classes
#

import pytest


import numpy as np
from scipy.sparse import coo_matrix

import pybamm
import sympy

EMPTY_DOMAINS = {
    "primary": [],
    "secondary": [],
    "tertiary": [],
    "quaternary": [],
}


class TestBinaryOperators:
    def test_binary_operator(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        bin = pybamm.BinaryOperator("binary test", a, b)
        assert bin.children[0].name == a.name
        assert bin.children[1].name == b.name
        c = pybamm.Scalar(1)
        d = pybamm.Scalar(2)
        bin2 = pybamm.BinaryOperator("binary test", c, d)
        with pytest.raises(NotImplementedError):
            bin2.evaluate()
        with pytest.raises(NotImplementedError):
            bin2._binary_jac(a, b)

    def test_binary_operator_domains(self):
        # same domain
        a = pybamm.Symbol("a", domain=["negative electrode"])
        b = pybamm.Symbol("b", domain=["negative electrode"])
        bin1 = pybamm.BinaryOperator("binary test", a, b)
        assert bin1.domain == ["negative electrode"]
        # one empty domain
        c = pybamm.Symbol("c", domain=[])
        bin2 = pybamm.BinaryOperator("binary test", a, c)
        assert bin2.domain == ["negative electrode"]
        bin3 = pybamm.BinaryOperator("binary test", c, b)
        assert bin3.domain == ["negative electrode"]
        # mismatched domains
        d = pybamm.Symbol("d", domain=["positive electrode"])
        with pytest.raises(pybamm.DomainError):
            pybamm.BinaryOperator("binary test", a, d)

    def test_addition(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        summ = pybamm.Addition(a, b)
        assert summ.children[0].name == a.name
        assert summ.children[1].name == b.name

        # test simplifying
        summ2 = pybamm.Scalar(1) + pybamm.Scalar(3)
        assert summ2 == pybamm.Scalar(4)

    def test_addition_numpy_array(self):
        a = pybamm.Symbol("a")
        # test adding symbol and numpy array
        # converts numpy array to vector
        array = np.array([1, 2, 3])
        summ3 = pybamm.Addition(a, array)
        assert isinstance(summ3, pybamm.Addition)
        assert isinstance(summ3.children[0], pybamm.Symbol)
        assert isinstance(summ3.children[1], pybamm.Vector)

        summ4 = array + a
        assert isinstance(summ4.children[0], pybamm.Vector)

        # should error if numpy array is not 1D
        array = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="left must be a 1D array"):
            pybamm.Addition(array, a)
        with pytest.raises(ValueError, match="right must be a 1D array"):
            pybamm.Addition(a, array)

    def test_power(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        pow1 = pybamm.Power(a, b)
        assert pow1.name == "**"
        assert pow1.children[0].name == a.name
        assert pow1.children[1].name == b.name

        a = pybamm.Scalar(4)
        b = pybamm.Scalar(2)
        pow2 = pybamm.Power(a, b)
        assert pow2.evaluate() == 16

    def test_diff(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        y = np.array([5, 3])

        # power
        assert (a**b).diff(b).evaluate(y=y) == 5**3 * np.log(5)
        assert (a**b).diff(a).evaluate(y=y) == 3 * 5**2
        assert (a**b).diff(a**b).evaluate() == 1
        assert (a**a).diff(a).evaluate(y=y) == 5**5 * np.log(5) + 5 * 5**4
        assert (a**a).diff(b).evaluate(y=y) == 0

        # addition
        assert (a + b).diff(a).evaluate() == 1
        assert (a + b).diff(b).evaluate() == 1
        assert (a + b).diff(a + b).evaluate() == 1
        assert (a + a).diff(a).evaluate() == 2
        assert (a + a).diff(b).evaluate() == 0

        # subtraction
        assert (a - b).diff(a).evaluate() == 1
        assert (a - b).diff(b).evaluate() == -1
        assert (a - b).diff(a - b).evaluate() == 1
        assert (a - a).diff(a).evaluate() == 0
        assert (a + a).diff(b).evaluate() == 0

        # multiplication
        assert (a * b).diff(a).evaluate(y=y) == 3
        assert (a * b).diff(b).evaluate(y=y) == 5
        assert (a * b).diff(a * b).evaluate(y=y) == 1
        assert (a * a).diff(a).evaluate(y=y) == 10
        assert (a * a).diff(b).evaluate(y=y) == 0

        # matrix multiplication (not implemented)
        matmul = a @ b
        with pytest.raises(NotImplementedError):
            matmul.diff(a)

        # inner
        assert pybamm.inner(a, b).diff(a).evaluate(y=y) == 3
        assert pybamm.inner(a, b).diff(b).evaluate(y=y) == 5
        assert pybamm.inner(a, b).diff(pybamm.inner(a, b)).evaluate(y=y) == 1
        assert pybamm.inner(a, a).diff(a).evaluate(y=y) == 10
        assert pybamm.inner(a, a).diff(b).evaluate(y=y) == 0

        # division
        assert (a / b).diff(a).evaluate(y=y) == 1 / 3
        assert (a / b).diff(b).evaluate(y=y) == -5 / 9
        assert (a / b).diff(a / b).evaluate(y=y) == 1
        assert (a / a).diff(a).evaluate(y=y) == 0
        assert (a / a).diff(b).evaluate(y=y) == 0

    def test_printing(self):
        # This in not an exhaustive list of all cases. More test cases may need to
        # be added for specific combinations of binary operators
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        assert str(a + b) == "a + b"
        assert str(a + b + c + d) == "a + b + c + d"
        assert str((a + b) + (c + d)) == "a + b + c + d"
        assert str(a + b - c) == "a + b - c"
        assert str(a + b - c + d) == "a + b - c + d"
        assert str((a + b) - (c + d)) == "a + b - (c + d)"
        assert str((a + b) - (c - d)) == "a + b - (c - d)"

        assert str((a + b) * (c + d)) == "(a + b) * (c + d)"
        assert str(a * b * (c + d)) == "a * b * (c + d)"
        assert str((a * b) * (c + d)) == "a * b * (c + d)"
        assert str(a * (b * (c + d))) == "a * b * (c + d)"
        assert str((a + b) / (c + d)) == "(a + b) / (c + d)"
        assert str(a + b / (c + d)) == "a + b / (c + d)"
        assert str(a * b / (c + d)) == "a * b / (c + d)"
        assert str((a * b) / (c + d)) == "a * b / (c + d)"
        assert str(a * (b / (c + d))) == "a * b / (c + d)"

    def test_eq(self):
        a = pybamm.Scalar(4)
        b = pybamm.Scalar(5)
        bin1 = pybamm.BinaryOperator("test", a, b)
        bin2 = pybamm.BinaryOperator("test", a, b)
        bin3 = pybamm.BinaryOperator("new test", a, b)
        assert bin1 == bin2
        assert bin1 != bin3
        c = pybamm.Scalar(5)
        bin4 = pybamm.BinaryOperator("test", a, c)
        assert bin1 == bin4
        d = pybamm.Scalar(42)
        bin5 = pybamm.BinaryOperator("test", a, d)
        assert bin1 != bin5

    def test_number_overloading(self):
        a = pybamm.Scalar(4)
        prod = a * 3
        assert isinstance(prod, pybamm.Scalar)
        assert prod.evaluate() == 12

    def test_sparse_multiply(self):
        row = np.array([0, 3, 1, 0])
        col = np.array([0, 3, 1, 2])
        data = np.array([4, 5, 7, 9])
        S1 = coo_matrix((data, (row, col)), shape=(4, 5))
        S2 = coo_matrix((data, (row, col)), shape=(5, 4))
        pybammS1 = pybamm.Matrix(S1)
        pybammS2 = pybamm.Matrix(S2)
        D1 = np.ones((4, 5))
        D2 = np.ones((5, 4))
        pybammD1 = pybamm.Matrix(D1)
        pybammD2 = pybamm.Matrix(D2)

        # Multiplication is elementwise
        np.testing.assert_array_equal(
            (pybammS1 * pybammS1).evaluate().toarray(), S1.multiply(S1).toarray()
        )
        np.testing.assert_array_equal(
            (pybammS2 * pybammS2).evaluate().toarray(), S2.multiply(S2).toarray()
        )
        np.testing.assert_array_equal(
            (pybammD1 * pybammS1).evaluate().toarray(), S1.toarray() * D1
        )
        np.testing.assert_array_equal(
            (pybammS1 * pybammD1).evaluate().toarray(), S1.toarray() * D1
        )
        np.testing.assert_array_equal(
            (pybammD2 * pybammS2).evaluate().toarray(), S2.toarray() * D2
        )
        np.testing.assert_array_equal(
            (pybammS2 * pybammD2).evaluate().toarray(), S2.toarray() * D2
        )
        with pytest.raises(pybamm.ShapeError, match="inconsistent shapes"):
            (pybammS1 * pybammS2).test_shape()
        with pytest.raises(pybamm.ShapeError, match="inconsistent shapes"):
            (pybammS2 * pybammS1).test_shape()
        with pytest.raises(pybamm.ShapeError, match="inconsistent shapes"):
            (pybammS2 * pybammS1).evaluate_ignoring_errors()

        # Matrix multiplication is normal matrix multiplication
        np.testing.assert_array_equal(
            (pybammS1 @ pybammS2).evaluate().toarray(), (S1 * S2).toarray()
        )
        np.testing.assert_array_equal(
            (pybammS2 @ pybammS1).evaluate().toarray(), (S2 * S1).toarray()
        )
        np.testing.assert_array_equal((pybammS1 @ pybammD2).evaluate(), S1 * D2)
        np.testing.assert_array_equal((pybammD2 @ pybammS1).evaluate(), D2 * S1)
        np.testing.assert_array_equal((pybammS2 @ pybammD1).evaluate(), S2 * D1)
        np.testing.assert_array_equal((pybammD1 @ pybammS2).evaluate(), D1 * S2)
        with pytest.raises(pybamm.ShapeError, match="dimension mismatch"):
            (pybammS1 @ pybammS1).test_shape()
        with pytest.raises(pybamm.ShapeError, match="dimension mismatch"):
            (pybammS2 @ pybammS2).test_shape()

    def test_sparse_divide(self):
        row = np.array([0, 3, 1, 0])
        col = np.array([0, 3, 1, 2])
        data = np.array([4, 5, 7, 9])
        S1 = coo_matrix((data, (row, col)), shape=(4, 5))
        pybammS1 = pybamm.Matrix(S1)
        v1 = np.ones((4, 1))
        pybammv1 = pybamm.Vector(v1)

        np.testing.assert_array_equal(
            (pybammS1 / pybammv1).evaluate().toarray(), S1.toarray() / v1
        )

    def test_inner(self):
        model = pybamm.lithium_ion.BaseModel()

        phi_s = pybamm.Variable(
            "Negative electrode potential [V]", domain="negative electrode"
        )
        i = pybamm.grad(phi_s)

        model.rhs = {phi_s: pybamm.inner(i, i)}
        model.boundary_conditions = {
            phi_s: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        model.initial_conditions = {phi_s: pybamm.Scalar(0)}

        model.variables = {"inner": pybamm.inner(i, i)}

        # load parameter values and process model and geometry
        param = model.default_parameter_values
        geometry = model.default_geometry
        param.process_model(model)
        param.process_geometry(geometry)

        # set mesh
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # check doesn't evaluate on edges anymore
        assert model.variables["inner"].evaluates_on_edges("primary") == False

    def test_source(self):
        u = pybamm.Variable("u", domain="current collector")
        v = pybamm.Variable("v", domain="current collector")

        source = pybamm.source(u, v)
        assert isinstance(source.children[0], pybamm.Mass)
        boundary_source = pybamm.source(u, v, boundary=True)
        assert isinstance(boundary_source.children[0], pybamm.BoundaryMass)

    def test_source_error(self):
        # test error with domain not current collector
        v = pybamm.Vector(np.ones(5), domain="current collector")
        w = pybamm.Vector(2 * np.ones(3), domain="test")
        with pytest.raises(pybamm.DomainError, match="'source'"):
            pybamm.source(v, w)

    def test_heaviside(self):
        b = pybamm.StateVector(slice(0, 1))
        heav = 1 < b
        assert heav.evaluate(y=np.array([2])) == 1
        assert heav.evaluate(y=np.array([1])) == 0
        assert heav.evaluate(y=np.array([0])) == 0
        assert str(heav) == "1.0 < y[0:1]"

        heav = 1 >= b
        assert heav.evaluate(y=np.array([2])) == 0
        assert heav.evaluate(y=np.array([1])) == 1
        assert heav.evaluate(y=np.array([0])) == 1
        assert str(heav) == "y[0:1] <= 1.0"

        # simplifications
        assert (1 < b + 2) == (-1 < b)
        assert (b + 1 > 2) == (b > 1)

        # expression with a subtract
        expr = 2 * (b < 1) - (b > 3)
        assert expr.evaluate(y=np.array([0])) == 2
        assert expr.evaluate(y=np.array([2])) == 0
        assert expr.evaluate(y=np.array([4])) == -1

    def test_equality(self):
        a = pybamm.Scalar(1)
        b = pybamm.StateVector(slice(0, 1))
        equal = pybamm.Equality(a, b)
        assert equal.evaluate(y=np.array([1])) == 1
        assert equal.evaluate(y=np.array([2])) == 0
        assert str(equal) == "1.0 == y[0:1]"
        assert equal.diff(b) == 0

    def test_sigmoid(self):
        a = pybamm.Scalar(1)
        b = pybamm.StateVector(slice(0, 1))
        sigm = pybamm.sigmoid(a, b, 10)
        assert sigm.evaluate(y=np.array([2]))[0, 0] == pytest.approx(1)
        assert sigm.evaluate(y=np.array([1])) == 0.5
        pytest.approx(sigm.evaluate(y=np.array([0]))[0, 0], abs=0)
        assert str(sigm) == "0.5 + 0.5 * tanh(-10.0 + 10.0 * y[0:1])"

        sigm = pybamm.sigmoid(b, a, 10)
        pytest.approx(sigm.evaluate(y=np.array([2]))[0, 0], abs=0)
        assert sigm.evaluate(y=np.array([1])) == 0.5
        pytest.approx(sigm.evaluate(y=np.array([0]))[0, 0], abs=1)
        assert str(sigm) == "0.5 + 0.5 * tanh(10.0 - (10.0 * y[0:1]))"

    def test_modulo(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.Scalar(3)
        mod = a % b
        assert mod.evaluate(y=np.array([4]))[0, 0] == 1
        assert mod.evaluate(y=np.array([3]))[0, 0] == 0
        assert mod.evaluate(y=np.array([2]))[0, 0] == 2
        assert mod.evaluate(y=np.array([4.3]))[0, 0] == pytest.approx(1.3)
        assert mod.evaluate(y=np.array([2.2]))[0, 0] == pytest.approx(2.2)
        assert str(mod) == "y[0:1] mod 3.0"

    def test_minimum_maximum(self):
        a = pybamm.Scalar(1)
        b = pybamm.StateVector(slice(0, 1))
        minimum = pybamm.minimum(a, b)
        assert minimum.evaluate(y=np.array([2])) == 1
        assert minimum.evaluate(y=np.array([1])) == 1
        assert minimum.evaluate(y=np.array([0])) == 0
        assert str(minimum) == "minimum(1.0, y[0:1])"

        maximum = pybamm.maximum(a, b)
        assert maximum.evaluate(y=np.array([2])) == 2
        assert maximum.evaluate(y=np.array([1])) == 1
        assert maximum.evaluate(y=np.array([0])) == 1
        assert str(maximum) == "maximum(1.0, y[0:1])"

    def test_softminus_softplus(self):
        a = pybamm.Scalar(1)
        b = pybamm.StateVector(slice(0, 1))

        minimum = pybamm.softminus(a, b, 50)
        assert minimum.evaluate(y=np.array([2]))[0, 0] == pytest.approx(1)
        assert minimum.evaluate(y=np.array([0]))[0, 0] == pytest.approx(0)
        assert str(minimum) == "-0.02 * log(1.9287498479639178e-22 + exp(-50.0 * y[0:1]))"

        maximum = pybamm.softplus(a, b, 50)
        assert maximum.evaluate(y=np.array([2]))[0, 0] == pytest.approx(2)
        assert maximum.evaluate(y=np.array([0]))[0, 0] == pytest.approx(1)
        assert str(maximum)[:20] == \
            "0.02 * log(5.184705528587072e+21 + exp(50.0 * y[0:1]))"[:20]
        assert str(maximum)[-20:] == \
            "0.02 * log(5.184705528587072e+21 + exp(50.0 * y[0:1]))"[-20:]

        # Test that smooth min/max are used when the setting is changed
        pybamm.settings.min_max_mode = "soft"
        pybamm.settings.min_max_smoothing = 10

        assert str(pybamm.minimum(a, b)) == str(pybamm.softminus(a, b, 10))
        assert str(pybamm.maximum(a, b)) == str(pybamm.softplus(a, b, 10))

        # But exact min/max should still be used if both variables are constant
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        assert str(pybamm.minimum(a, b)) == str(a)
        assert str(pybamm.maximum(a, b)) == str(b)

        # Change setting back for other tests
        pybamm.settings.set_smoothing_parameters("exact")

    def test_smooth_minus_plus(self):
        a = pybamm.Scalar(1)
        b = pybamm.StateVector(slice(0, 1))

        minimum = pybamm.smooth_min(a, b, 3000)
        pytest.approx(minimum.evaluate(y=np.array([2]))[0, 0], abs=1)
        pytest.approx(minimum.evaluate(y=np.array([0]))[0, 0], abs=0)

        maximum = pybamm.smooth_max(a, b, 3000)
        assert maximum.evaluate(y=np.array([2]))[0, 0] == pytest.approx(2)
        assert maximum.evaluate(y=np.array([0]))[0, 0] == pytest.approx(1)

        minimum = pybamm.smooth_min(a, b, 1)
        assert str(minimum) == \
            "0.5 * (1.0 + y[0:1] - sqrt(1.0 + (1.0 - y[0:1]) ** 2.0))"
        maximum = pybamm.smooth_max(a, b, 1)
        assert str(maximum) == \
            "0.5 * (sqrt(1.0 + (1.0 - y[0:1]) ** 2.0) + 1.0 + y[0:1])"

        # Test that smooth min/max are used when the setting is changed
        pybamm.settings.min_max_mode = "smooth"

        pybamm.settings.min_max_smoothing = 1
        assert str(pybamm.minimum(a, b)) == str(pybamm.smooth_min(a, b, 1))
        assert str(pybamm.maximum(a, b)) == str(pybamm.smooth_max(a, b, 1))

        pybamm.settings.min_max_smoothing = 3000
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        assert str(pybamm.minimum(a, b)) == str(a)
        assert str(pybamm.maximum(a, b)) == str(b)

        # Change setting back for other tests
        pybamm.settings.set_smoothing_parameters("exact")

    def test_binary_simplifications(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        v = pybamm.Vector(np.zeros((10, 1)))
        v1 = pybamm.Vector(np.ones((10, 1)))
        f = pybamm.StateVector(slice(0, 10))

        var = pybamm.Variable("var", domain="domain")
        broad0 = pybamm.PrimaryBroadcast(0, "domain")
        broad1 = pybamm.PrimaryBroadcast(1, "domain")
        broad2 = pybamm.PrimaryBroadcast(2, "domain")
        broad2_edge = pybamm.PrimaryBroadcastToEdges(2, "domain")

        # power
        assert (c**0) == pybamm.Scalar(1)
        assert (0**c) == pybamm.Scalar(0)
        assert (c**1) == c
        # power with broadcasts
        assert (c**broad2) == pybamm.PrimaryBroadcast(c**2, "domain")
        assert (broad2**c) == pybamm.PrimaryBroadcast(2**c, "domain")
        assert (broad2 ** pybamm.PrimaryBroadcast(c, "domain")) == \
            pybamm.PrimaryBroadcast(2**c, "domain")
        # power with broadcasts to edge
        assert isinstance(var**broad2_edge, pybamm.Power)
        assert (var**broad2_edge).left == var
        assert (var**broad2_edge).right == broad2_edge

        # addition
        assert a + b == pybamm.Scalar(1)
        assert b + b == pybamm.Scalar(2)
        assert b + a == pybamm.Scalar(1)
        assert 0 + b == pybamm.Scalar(1)
        assert 0 + c == c
        assert c + 0 == c
        # addition with subtraction
        assert c + (d - c) == d
        assert (c - d) + d == c
        # addition with broadcast zero
        assert isinstance((1 + broad0), pybamm.PrimaryBroadcast)
        np.testing.assert_array_equal((1 + broad0).child.evaluate(), 1)
        np.testing.assert_array_equal((1 + broad0).domain, "domain")
        assert isinstance((broad0 + 1), pybamm.PrimaryBroadcast)
        np.testing.assert_array_equal((broad0 + 1).child.evaluate(), 1)
        np.testing.assert_array_equal((broad0 + 1).domain, "domain")
        # addition with broadcasts
        assert (c + broad2) == pybamm.PrimaryBroadcast(c + 2, "domain")
        assert (broad2 + c) == pybamm.PrimaryBroadcast(2 + c, "domain")
        # addition with negate
        assert c + -d == c - d
        assert -c + d == d - c

        # subtraction
        assert a - b == pybamm.Scalar(-1)
        assert b - b == pybamm.Scalar(0)
        assert b - a == pybamm.Scalar(1)
        # subtraction with addition
        assert c - (d + c) == -d
        assert c - (c - d) == d
        assert (c + d) - d == c
        assert (d + c) - d == c
        assert (d - c) - d == -c
        # subtraction with broadcasts
        assert (c - broad2) == pybamm.PrimaryBroadcast(c - 2, "domain")
        assert (broad2 - c) == pybamm.PrimaryBroadcast(2 - c, "domain")
        # subtraction from itself
        assert (c - c) == pybamm.Scalar(0)
        assert (broad2 - broad2) == broad0
        # subtraction with negate
        assert (c - (-d)) == c + d

        # addition and subtraction with matrix zero
        assert b + v == pybamm.Vector(np.ones((10, 1)))
        assert v + b == pybamm.Vector(np.ones((10, 1)))
        assert b - v == pybamm.Vector(np.ones((10, 1)))
        assert v - b == pybamm.Vector(-np.ones((10, 1)))

        # multiplication
        assert a * b == pybamm.Scalar(0)
        assert b * a == pybamm.Scalar(0)
        assert b * b == pybamm.Scalar(1)
        assert a * a == pybamm.Scalar(0)
        assert a * c == pybamm.Scalar(0)
        assert c * a == pybamm.Scalar(0)
        assert b * c == c
        # multiplication with -1
        assert (c * -1) == (-c)
        assert (-1 * c) == (-c)
        # multiplication with a negation
        assert (-c * -f) == (c * f)
        assert (-c * 4) == (c * -4)
        assert (4 * -c) == (-4 * c)
        # multiplication with division
        assert (c * (d / c)) == d
        assert (c / d) * d == c
        # multiplication with broadcasts
        assert (c * broad2) == pybamm.PrimaryBroadcast(c * 2, "domain")
        assert (broad2 * c) == pybamm.PrimaryBroadcast(2 * c, "domain")

        # multiplication with matrix zero
        assert b * v == pybamm.Vector(np.zeros((10, 1)))
        assert v * b == pybamm.Vector(np.zeros((10, 1)))
        # multiplication with matrix one
        assert (f * v1) == f
        assert (v1 * f) == f
        # multiplication with matrix minus one
        assert (f * (-v1)) == (-f)
        assert ((-v1) * f) == (-f)
        # multiplication with broadcast
        assert (var * broad2) == (var * 2)
        assert (broad2 * var) == (2 * var)
        # multiplication with broadcast one
        assert (var * broad1) == var
        assert (broad1 * var) == var
        # multiplication with broadcast minus one
        assert (var * -broad1) == (-var)
        assert (-broad1 * var) == (-var)

        # division by itself
        assert (c / c) == pybamm.Scalar(1)
        assert (broad2 / broad2) == broad1
        # division with a negation
        assert (-c / -f) == (c / f)
        assert (-c / 4) == -0.25 * c
        assert (4 / -c) == (-4 / c)
        # division with multiplication
        assert (c * d) / c == d
        assert (d * c) / c == d
        # division with broadcasts
        assert (c / broad2) == pybamm.PrimaryBroadcast(c / 2, "domain")
        assert (broad2 / c) == pybamm.PrimaryBroadcast(2 / c, "domain")
        # division with matrix one
        assert (f / v1) == f
        assert (f / -v1) == (-f)
        # division by zero
        with pytest.raises(ZeroDivisionError):
            b / a

        # division with a common term
        assert (2 * c) / (2 * var) == (c / var)
        assert (c * 2) / (var * 2) == (c / var)

    def test_binary_simplifications_concatenations(self):
        def conc_broad(x, y, z):
            return pybamm.concatenation(
                pybamm.PrimaryBroadcast(x, "negative electrode"),
                pybamm.PrimaryBroadcast(y, "separator"),
                pybamm.PrimaryBroadcast(z, "positive electrode"),
            )

        # Test that concatenations get simplified correctly
        a = conc_broad(1, 2, 3)
        b = conc_broad(11, 12, 13)
        c = conc_broad(
            pybamm.InputParameter("x"),
            pybamm.InputParameter("y"),
            pybamm.InputParameter("z"),
        )
        assert (a + 4) == conc_broad(5, 6, 7)
        assert (4 + a) == conc_broad(5, 6, 7)
        assert (a + b) == conc_broad(12, 14, 16)
        assert isinstance((a + c), pybamm.Concatenation)

        # No simplifications if all are Variable or StateVector objects
        v = pybamm.concatenation(
            pybamm.Variable("x", "negative electrode"),
            pybamm.Variable("y", "separator"),
            pybamm.Variable("z", "positive electrode"),
        )
        assert isinstance((v * v), pybamm.Multiplication)
        assert isinstance((a * v), pybamm.Multiplication)

    def test_advanced_binary_simplifications(self):
        # MatMul simplifications that often appear when discretising spatial operators
        A = pybamm.Matrix(np.random.rand(10, 10))
        B = pybamm.Matrix(np.random.rand(10, 10))
        C = pybamm.Matrix(np.random.rand(10, 10))
        var = pybamm.StateVector(slice(0, 10))
        var2 = pybamm.StateVector(slice(10, 20))
        vec = pybamm.Vector(np.random.rand(10))

        # Do A@B first if it is constant
        expr = A @ (B @ var)
        assert expr == ((A @ B) @ var)

        # Distribute the @ operator to a sum if one of the symbols being summed is
        # constant
        expr = A @ (var + vec)
        assert expr == ((A @ var) + (A @ vec))
        expr = A @ (var - vec)
        assert expr == ((A @ var) - (A @ vec))

        expr = A @ ((B @ var) + vec)
        assert expr == (((A @ B) @ var) + (A @ vec))
        expr = A @ ((B @ var) - vec)
        assert expr == (((A @ B) @ var) - (A @ vec))

        # Distribute the @ operator to a sum if both symbols being summed are matmuls
        expr = A @ (B @ var + C @ var2)
        assert expr == ((A @ B) @ var + (A @ C) @ var2)
        expr = A @ (B @ var - C @ var2)
        assert expr == ((A @ B) @ var - (A @ C) @ var2)

        # Reduce (A@var + B@var) to ((A+B)@var)
        expr = A @ var + B @ var
        assert expr == ((A + B) @ var)

        # Do A*e first if it is constant
        expr = A @ (5 * var)
        assert expr == ((A * 5) @ var)
        expr = A @ (var * 5)
        assert expr == ((A * 5) @ var)
        # Do A/e first if it is constant
        expr = A @ (var / 2)
        assert expr == ((A / 2) @ var)
        # Do (vec*A) first if it is constant
        expr = vec * (A @ var)
        assert expr == ((vec * A) @ var)
        expr = (A @ var) * vec
        assert expr == ((vec * A) @ var)
        # Do (A/vec) first if it is constant
        expr = (A @ var) / vec
        assert isinstance(expr, pybamm.MatrixMultiplication)
        np.testing.assert_array_almost_equal(expr.left.evaluate(), (A / vec).evaluate())
        assert expr.children[1] == var

        # simplify additions and subtractions
        expr = 7 + (var + 5)
        assert expr == (12 + var)
        expr = 7 + (5 + var)
        assert expr == (12 + var)
        expr = (var + 5) + 7
        assert expr == (var + 12)
        expr = (5 + var) + 7
        assert expr == (12 + var)
        expr = 7 + (var - 5)
        assert expr == (2 + var)
        expr = 7 + (5 - var)
        assert expr == (12 - var)
        expr = (var - 5) + 7
        assert expr == (var + 2)
        expr = (5 - var) + 7
        assert expr == (12 - var)
        expr = 7 - (var + 5)
        assert expr == (2 - var)
        expr = 7 - (5 + var)
        assert expr == (2 - var)
        expr = (var + 5) - 7
        assert expr == (var + -2)
        expr = (5 + var) - 7
        assert expr == (-2 + var)
        expr = 7 - (var - 5)
        assert expr == (12 - var)
        expr = 7 - (5 - var)
        assert expr == (2 + var)
        expr = (var - 5) - 7
        assert expr == (var - 12)
        expr = (5 - var) - 7
        assert expr == (-2 - var)
        expr = var - (var + var2)
        assert expr == -var2

        # simplify multiplications and divisions
        expr = 10 * (var * 5)
        assert expr == 50 * var
        expr = (var * 5) * 10
        assert expr == var * 50
        expr = 10 * (5 * var)
        assert expr == 50 * var
        expr = (5 * var) * 10
        assert expr == 50 * var
        expr = 10 * (var / 5)
        assert expr == (10 / 5) * var
        expr = (var / 5) * 10
        assert expr == var * (10 / 5)
        expr = (var * 5) / 10
        assert expr == var * (5 / 10)
        expr = (5 * var) / 10
        assert expr == (5 / 10) * var
        expr = 5 / (10 * var)
        assert expr == (5 / 10) / var
        expr = 5 / (var * 10)
        assert expr == (5 / 10) / var
        expr = (5 / var) / 10
        assert expr == (5 / 10) / var
        expr = 5 / (10 / var)
        assert expr == (5 / 10) * var
        expr = 5 / (var / 10)
        assert expr == 50 / var

        # use power rules on multiplications and divisions
        expr = (var * 5) ** 2
        assert expr == var**2 * 25
        expr = (5 * var) ** 2
        assert expr == 25 * var**2
        expr = (5 / var) ** 2
        assert expr == 25 / var**2

    def test_inner_simplifications(self):
        a1 = pybamm.Scalar(0)
        M1 = pybamm.Matrix(np.zeros((10, 10)))
        v1 = pybamm.Vector(np.ones(10))
        a2 = pybamm.Scalar(1)
        M2 = pybamm.Matrix(np.ones((10, 10)))
        a3 = pybamm.Scalar(3)

        np.testing.assert_array_equal(
            pybamm.inner(a1, M2).evaluate().toarray(), M1.entries
        )
        assert pybamm.inner(a1, a2).evaluate() == 0
        np.testing.assert_array_equal(
            pybamm.inner(M2, a1).evaluate().toarray(), M1.entries
        )
        assert pybamm.inner(a2, a1).evaluate() == 0
        np.testing.assert_array_equal(
            pybamm.inner(M1, a3).evaluate().toarray(), M1.entries
        )
        np.testing.assert_array_equal(pybamm.inner(v1, a3).evaluate(), 3 * v1.entries)
        assert pybamm.inner(a2, a3).evaluate() == 3
        assert pybamm.inner(a3, a2).evaluate() == 3
        assert pybamm.inner(a3, a3).evaluate() == 9

    def test_to_equation(self):
        # Test print_name
        pybamm.Addition.print_name = "test"
        assert pybamm.Addition(1, 2).to_equation() == sympy.Symbol("test")

        # Test Power
        assert pybamm.Power(7, 2).to_equation() == 49

        # Test Division
        assert pybamm.Division(10, 2).to_equation() == 5

        # Test Matrix Multiplication
        arr1 = pybamm.Array([[1, 0], [0, 1]])
        arr2 = pybamm.Array([[4, 1], [2, 2]])
        assert pybamm.MatrixMultiplication(arr1, arr2).to_equation() == \
            sympy.Matrix([[4.0, 1.0], [2.0, 2.0]])

        # Test EqualHeaviside
        assert pybamm.EqualHeaviside(1, 0).to_equation() == False

        # Test NotEqualHeaviside
        assert pybamm.NotEqualHeaviside(2, 4).to_equation() == True

    def test_to_json(self, mocker):
        # Test Addition
        add_json = {
            "name": "+",
            "id": mocker.ANY,
            "domains": EMPTY_DOMAINS,
        }
        add = pybamm.Addition(2, 4)

        assert add.to_json() == add_json

        add_json["children"] = [pybamm.Scalar(2), pybamm.Scalar(4)]
        assert pybamm.Addition._from_json(add_json) == add

        # Test Power
        pow_json = {
            "name": "**",
            "id": mocker.ANY,
            "domains": EMPTY_DOMAINS,
        }

        pow = pybamm.Power(7, 2)
        assert pow.to_json() == pow_json

        pow_json["children"] = [pybamm.Scalar(7), pybamm.Scalar(2)]
        assert pybamm.Power._from_json(pow_json) == pow

        # Test Division
        div_json = {
            "name": "/",
            "id": mocker.ANY,
            "domains": EMPTY_DOMAINS,
        }

        div = pybamm.Division(10, 5)
        assert div.to_json() == div_json

        div_json["children"] = [pybamm.Scalar(10), pybamm.Scalar(5)]
        assert pybamm.Division._from_json(div_json) == div

        # Test EqualHeaviside
        equal_json = {
            "name": "<=",
            "id": mocker.ANY,
            "domains": EMPTY_DOMAINS,
        }

        equal_h = pybamm.EqualHeaviside(2, 4)
        assert equal_h.to_json() == equal_json

        equal_json["children"] = [pybamm.Scalar(2), pybamm.Scalar(4)]
        assert pybamm.EqualHeaviside._from_json(equal_json) == equal_h

        # Test notEqualHeaviside
        not_equal_json = {
            "name": "<",
            "id": mocker.ANY,
            "domains": EMPTY_DOMAINS,
        }

        ne_h = pybamm.NotEqualHeaviside(2, 4)
        assert ne_h.to_json() == not_equal_json

        not_equal_json["children"] = [pybamm.Scalar(2), pybamm.Scalar(4)]
        assert pybamm.NotEqualHeaviside._from_json(not_equal_json) == ne_h
