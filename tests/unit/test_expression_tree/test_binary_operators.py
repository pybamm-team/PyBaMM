#
# Tests for the Binary Operator classes
#
import unittest

import numpy as np
import sympy
from scipy.sparse.coo import coo_matrix

import pybamm


class TestBinaryOperators(unittest.TestCase):
    def test_binary_operator(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        bin = pybamm.BinaryOperator("binary test", a, b)
        self.assertEqual(bin.children[0].name, a.name)
        self.assertEqual(bin.children[1].name, b.name)
        c = pybamm.Scalar(1)
        d = pybamm.Scalar(2)
        bin2 = pybamm.BinaryOperator("binary test", c, d)
        with self.assertRaises(NotImplementedError):
            bin2.evaluate()

    def test_binary_operator_domains(self):
        # same domain
        a = pybamm.Symbol("a", domain=["negative electrode"])
        b = pybamm.Symbol("b", domain=["negative electrode"])
        bin1 = pybamm.BinaryOperator("binary test", a, b)
        self.assertEqual(bin1.domain, ["negative electrode"])
        # one empty domain
        c = pybamm.Symbol("c", domain=[])
        bin2 = pybamm.BinaryOperator("binary test", a, c)
        self.assertEqual(bin2.domain, ["negative electrode"])
        bin3 = pybamm.BinaryOperator("binary test", c, b)
        self.assertEqual(bin3.domain, ["negative electrode"])
        # mismatched domains
        d = pybamm.Symbol("d", domain=["positive electrode"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.BinaryOperator("binary test", a, d)

    def test_addition(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        summ = pybamm.Addition(a, b)
        self.assertEqual(summ.children[0].name, a.name)
        self.assertEqual(summ.children[1].name, b.name)

        # test simplifying
        summ2 = pybamm.Scalar(1) + pybamm.Scalar(3)
        self.assertEqual(summ2.id, pybamm.Scalar(4).id)

    def test_power(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        pow1 = pybamm.Power(a, b)
        self.assertEqual(pow1.name, "**")
        self.assertEqual(pow1.children[0].name, a.name)
        self.assertEqual(pow1.children[1].name, b.name)

        a = pybamm.Scalar(4)
        b = pybamm.Scalar(2)
        pow2 = pybamm.Power(a, b)
        self.assertEqual(pow2.evaluate(), 16)

    def test_known_eval(self):
        # Scalars
        a = pybamm.Scalar(4)
        b = pybamm.StateVector(slice(0, 1))
        expr = (a + b) - (a + b) * (a + b)
        value = expr.evaluate(y=np.array([2]))
        self.assertEqual(expr.evaluate(y=np.array([2]), known_evals={})[0], value)
        self.assertIn((a + b).id, expr.evaluate(y=np.array([2]), known_evals={})[1])
        self.assertEqual(
            expr.evaluate(y=np.array([2]), known_evals={})[1][(a + b).id], 6
        )

        # Matrices
        a = pybamm.Matrix(np.random.rand(5, 5))
        b = pybamm.StateVector(slice(0, 5))
        expr2 = (a @ b) - (a @ b) * (a @ b) + (a @ b)
        y_test = np.linspace(0, 1, 5)
        value = expr2.evaluate(y=y_test)
        np.testing.assert_array_equal(
            expr2.evaluate(y=y_test, known_evals={})[0], value
        )
        self.assertIn((a @ b).id, expr2.evaluate(y=y_test, known_evals={})[1])
        np.testing.assert_array_equal(
            expr2.evaluate(y=y_test, known_evals={})[1][(a @ b).id],
            (a @ b).evaluate(y=y_test),
        )

    def test_diff(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        y = np.array([5, 3])

        # power
        self.assertEqual((a ** b).diff(b).evaluate(y=y), 5 ** 3 * np.log(5))
        self.assertEqual((a ** b).diff(a).evaluate(y=y), 3 * 5 ** 2)
        self.assertEqual((a ** b).diff(a ** b).evaluate(), 1)
        self.assertEqual(
            (a ** a).diff(a).evaluate(y=y), 5 ** 5 * np.log(5) + 5 * 5 ** 4
        )
        self.assertEqual((a ** a).diff(b).evaluate(y=y), 0)

        # addition
        self.assertEqual((a + b).diff(a).evaluate(), 1)
        self.assertEqual((a + b).diff(b).evaluate(), 1)
        self.assertEqual((a + b).diff(a + b).evaluate(), 1)
        self.assertEqual((a + a).diff(a).evaluate(), 2)
        self.assertEqual((a + a).diff(b).evaluate(), 0)

        # subtraction
        self.assertEqual((a - b).diff(a).evaluate(), 1)
        self.assertEqual((a - b).diff(b).evaluate(), -1)
        self.assertEqual((a - b).diff(a - b).evaluate(), 1)
        self.assertEqual((a - a).diff(a).evaluate(), 0)
        self.assertEqual((a + a).diff(b).evaluate(), 0)

        # multiplication
        self.assertEqual((a * b).diff(a).evaluate(y=y), 3)
        self.assertEqual((a * b).diff(b).evaluate(y=y), 5)
        self.assertEqual((a * b).diff(a * b).evaluate(y=y), 1)
        self.assertEqual((a * a).diff(a).evaluate(y=y), 10)
        self.assertEqual((a * a).diff(b).evaluate(y=y), 0)

        # matrix multiplication (not implemented)
        matmul = a @ b
        with self.assertRaises(NotImplementedError):
            matmul.diff(a)

        # inner
        self.assertEqual(pybamm.inner(a, b).diff(a).evaluate(y=y), 3)
        self.assertEqual(pybamm.inner(a, b).diff(b).evaluate(y=y), 5)
        self.assertEqual(pybamm.inner(a, b).diff(pybamm.inner(a, b)).evaluate(y=y), 1)
        self.assertEqual(pybamm.inner(a, a).diff(a).evaluate(y=y), 10)
        self.assertEqual(pybamm.inner(a, a).diff(b).evaluate(y=y), 0)

        # division
        self.assertEqual((a / b).diff(a).evaluate(y=y), 1 / 3)
        self.assertEqual((a / b).diff(b).evaluate(y=y), -5 / 9)
        self.assertEqual((a / b).diff(a / b).evaluate(y=y), 1)
        self.assertEqual((a / a).diff(a).evaluate(y=y), 0)
        self.assertEqual((a / a).diff(b).evaluate(y=y), 0)

    def test_printing(self):
        # This in not an exhaustive list of all cases. More test cases may need to
        # be added for specific combinations of binary operators
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        self.assertEqual(str(a + b), "a + b")
        self.assertEqual(str(a + b + c + d), "a + b + c + d")
        self.assertEqual(str((a + b) + (c + d)), "a + b + c + d")
        self.assertEqual(str(a + b - c), "a + b - c")
        self.assertEqual(str(a + b - c + d), "a + b - c + d")
        self.assertEqual(str((a + b) - (c + d)), "a + b - (c + d)")
        self.assertEqual(str((a + b) - (c - d)), "a + b - (c - d)")

        self.assertEqual(str((a + b) * (c + d)), "(a + b) * (c + d)")
        self.assertEqual(str(a * b * (c + d)), "a * b * (c + d)")
        self.assertEqual(str((a * b) * (c + d)), "a * b * (c + d)")
        self.assertEqual(str(a * (b * (c + d))), "a * b * (c + d)")
        self.assertEqual(str((a + b) / (c + d)), "(a + b) / (c + d)")
        self.assertEqual(str(a + b / (c + d)), "a + b / (c + d)")
        self.assertEqual(str(a * b / (c + d)), "a * b / (c + d)")
        self.assertEqual(str((a * b) / (c + d)), "a * b / (c + d)")
        self.assertEqual(str(a * (b / (c + d))), "a * b / (c + d)")

    def test_id(self):
        a = pybamm.Scalar(4)
        b = pybamm.Scalar(5)
        bin1 = pybamm.BinaryOperator("test", a, b)
        bin2 = pybamm.BinaryOperator("test", a, b)
        bin3 = pybamm.BinaryOperator("new test", a, b)
        self.assertEqual(bin1.id, bin2.id)
        self.assertNotEqual(bin1.id, bin3.id)
        c = pybamm.Scalar(5)
        bin4 = pybamm.BinaryOperator("test", a, c)
        self.assertEqual(bin1.id, bin4.id)
        d = pybamm.Scalar(42)
        bin5 = pybamm.BinaryOperator("test", a, d)
        self.assertNotEqual(bin1.id, bin5.id)

    def test_number_overloading(self):
        a = pybamm.Scalar(4)
        prod = a * 3
        self.assertIsInstance(prod, pybamm.Scalar)
        self.assertEqual(prod.evaluate(), 12)

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
        with self.assertRaisesRegex(pybamm.ShapeError, "inconsistent shapes"):
            (pybammS1 * pybammS2).test_shape()
        with self.assertRaisesRegex(pybamm.ShapeError, "inconsistent shapes"):
            (pybammS2 * pybammS1).test_shape()
        with self.assertRaisesRegex(pybamm.ShapeError, "inconsistent shapes"):
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
        with self.assertRaisesRegex(pybamm.ShapeError, "dimension mismatch"):
            (pybammS1 @ pybammS1).test_shape()
        with self.assertRaisesRegex(pybamm.ShapeError, "dimension mismatch"):
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

        phi_s = pybamm.standard_variables.phi_s_n
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
        self.assertEqual(model.variables["inner"].evaluates_on_edges("primary"), False)

    def test_source(self):
        u = pybamm.Variable("u", domain="current collector")
        v = pybamm.Variable("v", domain="current collector")

        source = pybamm.source(u, v)
        self.assertIsInstance(source.children[0], pybamm.Mass)
        boundary_source = pybamm.source(u, v, boundary=True)
        self.assertIsInstance(boundary_source.children[0], pybamm.BoundaryMass)

    def test_source_error(self):
        # test error with domain not current collector
        v = pybamm.Vector(np.ones(5), domain="current collector")
        w = pybamm.Vector(2 * np.ones(3), domain="test")
        with self.assertRaisesRegex(pybamm.DomainError, "'source'"):
            pybamm.source(v, w)

    def test_heaviside(self):
        a = pybamm.Scalar(1)
        b = pybamm.StateVector(slice(0, 1))
        heav = a < b
        self.assertEqual(heav.evaluate(y=np.array([2])), 1)
        self.assertEqual(heav.evaluate(y=np.array([1])), 0)
        self.assertEqual(heav.evaluate(y=np.array([0])), 0)
        self.assertEqual(str(heav), "1.0 < y[0:1]")

        heav = a >= b
        self.assertEqual(heav.evaluate(y=np.array([2])), 0)
        self.assertEqual(heav.evaluate(y=np.array([1])), 1)
        self.assertEqual(heav.evaluate(y=np.array([0])), 1)
        self.assertEqual(str(heav), "y[0:1] <= 1.0")

    def test_sigmoid(self):
        a = pybamm.Scalar(1)
        b = pybamm.StateVector(slice(0, 1))
        sigm = pybamm.sigmoid(a, b, 10)
        self.assertAlmostEqual(sigm.evaluate(y=np.array([2]))[0, 0], 1)
        self.assertEqual(sigm.evaluate(y=np.array([1])), 0.5)
        self.assertAlmostEqual(sigm.evaluate(y=np.array([0]))[0, 0], 0)
        self.assertEqual(str(sigm), "(1.0 + tanh(10.0 * (y[0:1] - 1.0))) / 2.0")

        sigm = pybamm.sigmoid(b, a, 10)
        self.assertAlmostEqual(sigm.evaluate(y=np.array([2]))[0, 0], 0)
        self.assertEqual(sigm.evaluate(y=np.array([1])), 0.5)
        self.assertAlmostEqual(sigm.evaluate(y=np.array([0]))[0, 0], 1)
        self.assertEqual(str(sigm), "(1.0 + tanh(10.0 * (1.0 - y[0:1]))) / 2.0")

    def test_modulo(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.Scalar(3)
        mod = a % b
        self.assertEqual(mod.evaluate(y=np.array([4]))[0, 0], 1)
        self.assertEqual(mod.evaluate(y=np.array([3]))[0, 0], 0)
        self.assertEqual(mod.evaluate(y=np.array([2]))[0, 0], 2)
        self.assertAlmostEqual(mod.evaluate(y=np.array([4.3]))[0, 0], 1.3)
        self.assertAlmostEqual(mod.evaluate(y=np.array([2.2]))[0, 0], 2.2)
        self.assertEqual(str(mod), "y[0:1] mod 3.0")

    def test_minimum_maximum(self):
        a = pybamm.Scalar(1)
        b = pybamm.StateVector(slice(0, 1))
        minimum = pybamm.minimum(a, b)
        self.assertEqual(minimum.evaluate(y=np.array([2])), 1)
        self.assertEqual(minimum.evaluate(y=np.array([1])), 1)
        self.assertEqual(minimum.evaluate(y=np.array([0])), 0)
        self.assertEqual(str(minimum), "minimum(1.0, y[0:1])")

        maximum = pybamm.maximum(a, b)
        self.assertEqual(maximum.evaluate(y=np.array([2])), 2)
        self.assertEqual(maximum.evaluate(y=np.array([1])), 1)
        self.assertEqual(maximum.evaluate(y=np.array([0])), 1)
        self.assertEqual(str(maximum), "maximum(1.0, y[0:1])")

    def test_softminus_softplus(self):
        a = pybamm.Scalar(1)
        b = pybamm.StateVector(slice(0, 1))

        minimum = pybamm.softminus(a, b, 50)
        self.assertAlmostEqual(minimum.evaluate(y=np.array([2]))[0, 0], 1)
        self.assertAlmostEqual(minimum.evaluate(y=np.array([0]))[0, 0], 0)
        self.assertEqual(
            str(minimum), "log(1.9287498479639178e-22 + exp(-50.0 * y[0:1])) / -50.0"
        )

        maximum = pybamm.softplus(a, b, 50)
        self.assertAlmostEqual(maximum.evaluate(y=np.array([2]))[0, 0], 2)
        self.assertAlmostEqual(maximum.evaluate(y=np.array([0]))[0, 0], 1)
        self.assertEqual(
            str(maximum), "log(5.184705528587072e+21 + exp(50.0 * y[0:1])) / 50.0"
        )

        # Test that smooth min/max are used when the setting is changed
        pybamm.settings.min_smoothing = 10
        pybamm.settings.max_smoothing = 10

        self.assertEqual(str(pybamm.minimum(a, b)), str(pybamm.softminus(a, b, 10)))
        self.assertEqual(str(pybamm.maximum(a, b)), str(pybamm.softplus(a, b, 10)))

        # But exact min/max should still be used if both variables are constant
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        self.assertEqual(str(pybamm.minimum(a, b)), str(a))
        self.assertEqual(str(pybamm.maximum(a, b)), str(b))

        # Change setting back for other tests
        pybamm.settings.min_smoothing = "exact"
        pybamm.settings.max_smoothing = "exact"

    def test_binary_simplifications(self):
        a = pybamm.Scalar(0, domain="domain")
        b = pybamm.Scalar(1)
        c = pybamm.Parameter("c")
        v = pybamm.Vector(np.zeros((10, 1)))
        v1 = pybamm.Vector(np.ones((10, 1)))
        f = pybamm.StateVector(slice(0, 10))

        var = pybamm.Variable("var", domain="domain")
        broad0 = pybamm.PrimaryBroadcast(0, "domain")
        broad1 = pybamm.PrimaryBroadcast(1, "domain")
        broad2 = pybamm.PrimaryBroadcast(2, "domain")
        broad2_edge = pybamm.PrimaryBroadcastToEdges(2, "domain")

        # power
        self.assertEqual((c ** a).id, pybamm.Scalar(1).id)
        self.assertEqual((0 ** c).id, pybamm.Scalar(0).id)
        self.assertEqual((c ** b).id, c.id)
        # power with broadcasts
        self.assertEqual((c ** broad2).id, pybamm.PrimaryBroadcast(c ** 2, "domain").id)
        self.assertEqual((broad2 ** c).id, pybamm.PrimaryBroadcast(2 ** c, "domain").id)
        self.assertEqual(
            (broad2 ** pybamm.PrimaryBroadcast(c, "domain")).id,
            pybamm.PrimaryBroadcast(2 ** c, "domain").id,
        )
        # power with broadcasts to edge
        self.assertIsInstance(var ** broad2_edge, pybamm.Power)
        self.assertEqual((var ** broad2_edge).left.id, var.id)
        self.assertEqual((var ** broad2_edge).right.id, broad2_edge.id)

        # addition
        self.assertIsInstance((a + b), pybamm.Scalar)
        self.assertEqual((a + b).evaluate(), 1)
        self.assertIsInstance((b + b), pybamm.Scalar)
        self.assertEqual((b + b).evaluate(), 2)
        self.assertIsInstance((b + a), pybamm.Scalar)
        self.assertEqual((b + a).evaluate(), 1)
        self.assertIsInstance((0 + b), pybamm.Scalar)
        self.assertEqual((0 + b).evaluate(), 1)
        self.assertIsInstance((a + c), pybamm.Parameter)
        self.assertIsInstance((c + a), pybamm.Parameter)
        self.assertIsInstance((c + b), pybamm.Addition)
        self.assertIsInstance((b + c), pybamm.Addition)
        # rearranging additions
        self.assertEqual(((c + 1) + 2).id, (c + 3).id)
        self.assertEqual(((1 + c) + 2).id, (3 + c).id)
        self.assertEqual((2 + (c + 1)).id, (3 + c).id)
        self.assertEqual((2 + (1 + c)).id, (3 + c).id)
        # addition with broadcast zero
        self.assertIsInstance((b + broad0), pybamm.PrimaryBroadcast)
        np.testing.assert_array_equal((b + broad0).child.evaluate(), 1)
        np.testing.assert_array_equal((b + broad0).domain, "domain")
        self.assertIsInstance((broad0 + b), pybamm.PrimaryBroadcast)
        np.testing.assert_array_equal((broad0 + b).child.evaluate(), 1)
        np.testing.assert_array_equal((broad0 + b).domain, "domain")
        # addition with broadcasts
        self.assertEqual((c + broad2).id, pybamm.PrimaryBroadcast(c + 2, "domain").id)
        self.assertEqual((broad2 + c).id, pybamm.PrimaryBroadcast(2 + c, "domain").id)

        # subtraction
        self.assertIsInstance((a - b), pybamm.Scalar)
        self.assertEqual((a - b).evaluate(), -1)
        self.assertIsInstance((b - b), pybamm.Scalar)
        self.assertEqual((b - b).evaluate(), 0)
        self.assertIsInstance((b - a), pybamm.Scalar)
        self.assertEqual((b - a).evaluate(), 1)
        # subtraction with broadcasts
        self.assertEqual((c - broad2).id, pybamm.PrimaryBroadcast(c - 2, "domain").id)
        self.assertEqual((broad2 - c).id, pybamm.PrimaryBroadcast(2 - c, "domain").id)
        # subtraction from itself
        self.assertEqual((c - c).id, pybamm.Scalar(0).id)
        self.assertEqual((broad2 - broad2).id, broad0.id)

        # addition and subtraction with matrix zero
        self.assertIsInstance((b + v), pybamm.Array)
        np.testing.assert_array_equal((b + v).evaluate(), np.ones((10, 1)))
        self.assertIsInstance((v + b), pybamm.Array)
        np.testing.assert_array_equal((v + b).evaluate(), np.ones((10, 1)))
        self.assertIsInstance((b - v), pybamm.Array)
        np.testing.assert_array_equal((b - v).evaluate(), np.ones((10, 1)))
        self.assertIsInstance((v - b), pybamm.Array)
        np.testing.assert_array_equal((v - b).evaluate(), -np.ones((10, 1)))

        # multiplication
        self.assertIsInstance((a * b), pybamm.Scalar)
        self.assertEqual((a * b).evaluate(), 0)
        self.assertIsInstance((b * a), pybamm.Scalar)
        self.assertEqual((b * a).evaluate(), 0)
        self.assertIsInstance((b * b), pybamm.Scalar)
        self.assertEqual((b * b).evaluate(), 1)
        self.assertIsInstance((a * a), pybamm.Scalar)
        self.assertEqual((a * a).evaluate(), 0)
        self.assertIsInstance((a * c), pybamm.Scalar)
        self.assertEqual((a * c).evaluate(), 0)
        self.assertIsInstance((c * a), pybamm.Scalar)
        self.assertEqual((c * a).evaluate(), 0)
        self.assertIsInstance((b * c), pybamm.Parameter)
        self.assertIsInstance((2 * c), pybamm.Multiplication)
        # multiplication with -1
        self.assertEqual((c * -1).id, (-c).id)
        self.assertEqual((-1 * c).id, (-c).id)
        # multiplication with a negation
        self.assertEqual((-c * -f).id, (c * f).id)
        self.assertEqual((-c * 4).id, (c * -4).id)
        self.assertEqual((4 * -c).id, (-4 * c).id)
        # multiplication with broadcasts
        self.assertEqual((c * broad2).id, pybamm.PrimaryBroadcast(c * 2, "domain").id)
        self.assertEqual((broad2 * c).id, pybamm.PrimaryBroadcast(2 * c, "domain").id)

        # multiplication with matrix zero
        self.assertIsInstance((b * v), pybamm.Array)
        np.testing.assert_array_equal((b * v).evaluate(), np.zeros((10, 1)))
        self.assertIsInstance((v * b), pybamm.Array)
        np.testing.assert_array_equal((v * b).evaluate(), np.zeros((10, 1)))
        # multiplication with matrix one
        self.assertEqual((f * v1), f)
        self.assertEqual((v1 * f), f)
        # multiplication with matrix minus one
        self.assertEqual((f * (-v1)).id, (-f).id)
        self.assertEqual(((-v1) * f).id, (-f).id)
        # multiplication with broadcast
        self.assertEqual((var * broad2).id, (var * 2).id)
        self.assertEqual((broad2 * var).id, (2 * var).id)
        # multiplication with broadcast one
        self.assertEqual((var * broad1).id, var.id)
        self.assertEqual((broad1 * var).id, var.id)
        # multiplication with broadcast minus one
        self.assertEqual((var * -broad1).id, (-var).id)
        self.assertEqual((-broad1 * var).id, (-var).id)

        # division by itself
        self.assertEqual((c / c).id, pybamm.Scalar(1).id)
        self.assertEqual((broad2 / broad2).id, broad1.id)
        # division with a negation
        self.assertEqual((-c / -f).id, (c / f).id)
        self.assertEqual((-c / 4).id, (c / -4).id)
        self.assertEqual((4 / -c).id, (-4 / c).id)
        # division with broadcasts
        self.assertEqual((c / broad2).id, pybamm.PrimaryBroadcast(c / 2, "domain").id)
        self.assertEqual((broad2 / c).id, pybamm.PrimaryBroadcast(2 / c, "domain").id)
        # division with matrix one
        self.assertEqual((f / v1), f)
        self.assertEqual((f / -v1).id, (-f).id)
        # division by zero
        with self.assertRaises(ZeroDivisionError):
            b / a

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
        self.assertEqual((a + 4).id, conc_broad(5, 6, 7).id)
        self.assertEqual((4 + a).id, conc_broad(5, 6, 7).id)
        self.assertEqual((a + b).id, conc_broad(12, 14, 16).id)
        self.assertIsInstance((a + c), pybamm.Concatenation)

        # No simplifications if are Variable or StateVector objects
        v = pybamm.concatenation(
            pybamm.Variable("x", "negative electrode"),
            pybamm.Variable("y", "separator"),
            pybamm.Variable("z", "positive electrode"),
        )
        self.assertIsInstance((v * v), pybamm.Multiplication)
        self.assertIsInstance((a * v), pybamm.Multiplication)

    def test_advanced_binary_simplifications(self):
        # MatMul simplifications that often appear when discretising spatial operators
        A = pybamm.Matrix(np.random.rand(10, 10))
        B = pybamm.Matrix(np.random.rand(10, 10))
        var = pybamm.StateVector(slice(0, 10))
        d = pybamm.Vector(np.random.rand(10))
        e = pybamm.Scalar(5)
        f = pybamm.Scalar(7)

        # Do A@B first if it is constant
        expr = A @ (B @ var)
        self.assertEqual(expr.id, ((A @ B) @ var).id)

        # Distribute the @ operator to a sum if one of the symbols being summed is
        # constant
        expr = A @ (var + d)
        self.assertEqual(expr.id, ((A @ var) + (A @ d)).id)

        expr = A @ ((B @ var) + d)
        self.assertEqual(expr.id, (((A @ B) @ var) + (A @ d)).id)

        # Reduce (A@var + B@var) to ((A+B)@var)
        expr = A @ var + B @ var
        self.assertEqual(expr.id, ((A + B) @ var).id)

        # Do A*e first if it is constant
        expr = A @ (e * var)
        self.assertEqual(expr.id, ((A * e) @ var).id)
        expr = A @ (var * e)
        self.assertEqual(expr.id, ((A * e) @ var).id)
        # Do A/e first if it is constant
        expr = A @ (var / e)
        self.assertEqual(expr.id, ((A / e) @ var).id)
        # Do (d*A) first if it is constant
        expr = d * (A @ var)
        self.assertEqual(expr.id, ((d * A) @ var).id)
        expr = (A @ var) * d
        self.assertEqual(expr.id, ((d * A) @ var).id)
        # Do (A/d) first if it is constant
        expr = (A @ var) / d
        self.assertEqual(expr.id, ((A / d) @ var).id)

        # simplify multiplications and divisions
        expr = f * (var * e)
        self.assertEqual(expr.id, ((f * e) * var).id)
        expr = (var * e) * f
        self.assertEqual(expr.id, (var * (e * f)).id)
        expr = f * (e * var)
        self.assertEqual(expr.id, ((f * e) * var).id)
        expr = (e * var) * f
        self.assertEqual(expr.id, ((e * f) * var).id)
        expr = f * (var / e)
        self.assertEqual(expr.id, ((f / e) * var).id)
        expr = (var / e) * f
        self.assertEqual(expr.id, (var * (f / e)).id)
        expr = (var * e) / f
        self.assertEqual(expr.id, (var * (e / f)).id)
        expr = (e * var) / f
        self.assertEqual(expr.id, ((e / f) * var).id)

        # use power rules on multiplications and divisions
        expr = (var * e) ** 2
        self.assertEqual(expr.id, (var ** 2 * e ** 2).id)
        expr = (e * var) ** 2
        self.assertEqual(expr.id, (e ** 2 * var ** 2).id)
        expr = (var / e) ** 2
        self.assertEqual(expr.id, (var ** 2 / e ** 2).id)
        expr = (e / var) ** 2
        self.assertEqual(expr.id, (e ** 2 / var ** 2).id)

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
        self.assertEqual(pybamm.inner(a1, a2).evaluate(), 0)
        np.testing.assert_array_equal(
            pybamm.inner(M2, a1).evaluate().toarray(), M1.entries
        )
        self.assertEqual(pybamm.inner(a2, a1).evaluate(), 0)
        np.testing.assert_array_equal(
            pybamm.inner(M1, a3).evaluate().toarray(), M1.entries
        )
        np.testing.assert_array_equal(pybamm.inner(v1, a3).evaluate(), 3 * v1.entries)
        self.assertEqual(pybamm.inner(a2, a3).evaluate(), 3)
        self.assertEqual(pybamm.inner(a3, a2).evaluate(), 3)
        self.assertEqual(pybamm.inner(a3, a3).evaluate(), 9)

    def test_to_equation(self):
        # Test print_name
        pybamm.Addition.print_name = "test"
        self.assertEqual(pybamm.Addition(1, 2).to_equation(), sympy.symbols("test"))

        # Test Power
        self.assertEqual(pybamm.Power(7, 2).to_equation(), 49)

        # Test Division
        self.assertEqual(pybamm.Division(10, 2).to_equation(), 5)

        # Test Matrix Multiplication
        arr1 = pybamm.Array([[1, 0], [0, 1]])
        arr2 = pybamm.Array([[4, 1], [2, 2]])
        self.assertEqual(
            pybamm.MatrixMultiplication(arr1, arr2).to_equation(),
            sympy.Matrix([[4.0, 1.0], [2.0, 2.0]]),
        )

        # Test EqualHeaviside
        self.assertEqual(pybamm.EqualHeaviside(1, 0).to_equation(), False)

        # Test NotEqualHeaviside
        self.assertEqual(pybamm.NotEqualHeaviside(2, 4).to_equation(), True)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
