#
# Tests for the Binary Operator classes
#
import pybamm

import numpy as np
import unittest
from scipy.sparse.coo import coo_matrix


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
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c")
        d = pybamm.Symbol("d")
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
            str(minimum),
            "log(1.9287498479639178e-22 + exp(-50.0 * y[0:1])) / -50.0",
        )

        maximum = pybamm.softplus(a, b, 50)
        self.assertAlmostEqual(maximum.evaluate(y=np.array([2]))[0, 0], 2)
        self.assertAlmostEqual(maximum.evaluate(y=np.array([0]))[0, 0], 1)
        self.assertEqual(
            str(maximum),
            "log(5.184705528587072e+21 + exp(50.0 * y[0:1])) / 50.0",
        )

        # Test that smooth min/max are used when the setting is changed
        pybamm.settings.min_smoothing = 10
        pybamm.settings.max_smoothing = 10

        self.assertEqual(str(pybamm.minimum(a, b)), str(pybamm.softminus(a, b, 10)))
        self.assertEqual(str(pybamm.maximum(a, b)), str(pybamm.softplus(a, b, 10)))

        # But exact min/max should still be used if both variables are constant
        a = 1
        b = pybamm.Parameter("b")
        self.assertEqual(str(pybamm.minimum(a, b)), str(pybamm.Minimum(a, b)))
        self.assertEqual(str(pybamm.maximum(a, b)), str(pybamm.Maximum(a, b)))

        # Change setting back for other tests
        pybamm.settings.min_smoothing = "exact"
        pybamm.settings.max_smoothing = "exact"


class TestIsZero(unittest.TestCase):
    def test_is_scalar_zero(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(2)
        self.assertTrue(pybamm.is_scalar_zero(a))
        self.assertFalse(pybamm.is_scalar_zero(b))

    def test_is_matrix_zero(self):
        a = pybamm.Matrix(coo_matrix(np.zeros((10, 10))))
        b = pybamm.Matrix(coo_matrix(np.ones((10, 10))))
        c = pybamm.Matrix(coo_matrix(([1], ([0], [0])), shape=(5, 5)))
        self.assertTrue(pybamm.is_matrix_zero(a))
        self.assertFalse(pybamm.is_matrix_zero(b))
        self.assertFalse(pybamm.is_matrix_zero(c))

        a = pybamm.Matrix(np.zeros((10, 10)))
        b = pybamm.Matrix(np.ones((10, 10)))
        c = pybamm.Matrix([1, 0, 0])
        self.assertTrue(pybamm.is_matrix_zero(a))
        self.assertFalse(pybamm.is_matrix_zero(b))
        self.assertFalse(pybamm.is_matrix_zero(c))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
