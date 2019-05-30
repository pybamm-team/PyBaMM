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
        sum = pybamm.Addition(a, b)
        self.assertEqual(sum.children[0].name, a.name)
        self.assertEqual(sum.children[1].name, b.name)

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

    def test_outer(self):
        # Outer class
        v = pybamm.Vector(np.ones(5), domain="current collector")
        w = pybamm.Vector(2 * np.ones(3), domain="test")
        outer = pybamm.Outer(v, w)
        np.testing.assert_array_equal(outer.evaluate(), 2 * np.ones((15, 1)))
        self.assertEqual(outer.domain, w.domain)

        # outer function
        # if there is no domain clash, normal multiplication is retured
        u = pybamm.Vector(np.linspace(0, 1, 5))
        outer = pybamm.outer(u, v)
        self.assertIsInstance(outer, pybamm.Multiplication)
        np.testing.assert_array_equal(outer.evaluate(), u.evaluate())
        # otherwise, Outer class is returned
        outer_fun = pybamm.outer(v, w)
        outer_class = pybamm.Outer(v, w)
        self.assertEqual(outer_fun.id, outer_class.id)

        # failures
        with self.assertRaisesRegex(
            pybamm.DomainError, "left child domain must be 'current collector'"
        ):
            pybamm.Outer(w, v)
        y = pybamm.StateVector(slice(10))
        with self.assertRaisesRegex(
            TypeError, "right child must only contain SpatialVariable and scalars"
        ):
            pybamm.Outer(v, y)
        with self.assertRaises(NotImplementedError):
            outer_fun.diff(None)
        with self.assertRaises(NotImplementedError):
            outer_fun.jac(None)

    def test_known_eval(self):
        # Scalars
        a = pybamm.Scalar(4)
        b = pybamm.Scalar(2)
        expr = (a + b) - (a + b) * (a + b)
        value = expr.evaluate()
        self.assertEqual(expr.evaluate(known_evals={})[0], value)
        self.assertIn((a + b).id, expr.evaluate(known_evals={})[1])
        self.assertEqual(expr.evaluate(known_evals={})[1][(a + b).id], 6)

        # Matrices
        a = pybamm.Matrix(np.random.rand(5, 5))
        b = pybamm.Matrix(np.random.rand(5, 5))
        expr2 = (a @ b) - (a @ b) * (a @ b) + (a @ b)
        value = expr2.evaluate()
        np.testing.assert_array_equal(expr2.evaluate(known_evals={})[0], value)
        self.assertIn((a @ b).id, expr2.evaluate(known_evals={})[1])
        np.testing.assert_array_equal(
            expr2.evaluate(known_evals={})[1][(a @ b).id], (a @ b).evaluate()
        )

        # Expect using known evals to be faster than not
        timer = pybamm.Timer()
        start = timer.time()
        for _ in range(2000):
            expr2.evaluate()
        end = timer.time()
        start_known_evals = timer.time()
        for _ in range(2000):
            expr2.evaluate(known_evals={})
        end_known_evals = timer.time()
        self.assertLess(end_known_evals - start_known_evals, 1.2 * (end - start))
        self.assertGreater(end - start, (end_known_evals - start_known_evals))

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

        # division
        self.assertEqual((a / b).diff(a).evaluate(y=y), 1 / 3)
        self.assertEqual((a / b).diff(b).evaluate(y=y), -5 / 9)
        self.assertEqual((a / b).diff(a / b).evaluate(y=y), 1)
        self.assertEqual((a / a).diff(a).evaluate(y=y), 0)
        self.assertEqual((a / a).diff(b).evaluate(y=y), 0)

    def test_addition_printing(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        sum = pybamm.Addition(a, b)
        self.assertEqual(sum.name, "+")
        self.assertEqual(str(sum), "a + b")

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
        self.assertIsInstance(prod.children[1], pybamm.Scalar)
        self.assertEqual(prod.evaluate(), 12)

    def test_sparse_multiply(self):
        pybamm.debug_mode = True
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
            pybammS1 * pybammS2
        with self.assertRaisesRegex(pybamm.ShapeError, "inconsistent shapes"):
            pybammS2 * pybammS1

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
            pybammS1 @ pybammS1
        with self.assertRaisesRegex(pybamm.ShapeError, "dimension mismatch"):
            pybammS2 @ pybammS2
        pybamm.debug_mode = False

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
        c = pybamm.Matrix(np.array([1, 0, 0]))
        self.assertTrue(pybamm.is_matrix_zero(a))
        self.assertFalse(pybamm.is_matrix_zero(b))
        self.assertFalse(pybamm.is_matrix_zero(c))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
