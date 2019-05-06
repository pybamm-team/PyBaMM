#
# Tests for the Matrix class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import numpy as np

import unittest


class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
        self.x = np.array([1, 2, 3])
        self.mat = pybamm.Matrix(self.A)
        self.vect = pybamm.Vector(self.x)

    def test_array_wrapper(self):
        self.assertEqual(self.mat.ndim, 2)
        self.assertEqual(self.mat.shape, (3, 3))
        self.assertEqual(self.mat.size, 9)

    def test_matrix_evaluate(self):
        np.testing.assert_array_equal(self.mat.evaluate(), self.A)

    def test_matrix_operations(self):
        np.testing.assert_array_equal((self.mat + self.mat).evaluate(), 2 * self.A)
        np.testing.assert_array_equal((self.mat - self.mat).evaluate(), 0 * self.A)
        np.testing.assert_array_equal(
            (self.mat @ self.vect).evaluate(), np.array([5, 2, 3])
        )

    def test_matrix_simplifications(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)
        c = pybamm.Parameter("c")

        # matrix multiplication
        A = pybamm.Matrix(np.array([[1, 0], [0, 1]]))
        self.assertIsInstance((a @ A).simplify(), pybamm.Scalar)
        self.assertEqual((a @ A).simplify().evaluate(), 0)
        self.assertIsInstance((A @ a).simplify(), pybamm.Scalar)
        self.assertEqual((A @ a).simplify().evaluate(), 0)

        self.assertIsInstance((A @ c).simplify(), pybamm.MatrixMultiplication)

        # matrix * matrix
        m1 = pybamm.Matrix(np.array([[2, 0], [0, 2]]))
        m2 = pybamm.Matrix(np.array([[3, 0], [0, 3]]))
        v = pybamm.StateVector(slice(0, 2))
        v2 = pybamm.StateVector(slice(2, 4))

        for expr in [((m2 @ m1) @ v).simplify(), (m2 @ (m1 @ v)).simplify()]:
            self.assertIsInstance(expr.children[0], pybamm.Matrix)
            self.assertIsInstance(expr.children[1], pybamm.StateVector)
            np.testing.assert_array_equal(
                expr.children[0].entries, np.array([[6, 0], [0, 6]])
            )

        # div by a constant
        for expr in [((m2 @ m1) @ v / 2).simplify(), (m2 @ (m1 @ v) / 2).simplify()]:
            self.assertIsInstance(expr.children[0], pybamm.Matrix)
            self.assertIsInstance(expr.children[1], pybamm.StateVector)
            np.testing.assert_array_equal(
                expr.children[0].entries, np.array([[3, 0], [0, 3]])
            )

        expr = (v @ v / 2).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 0.5)
        self.assertIsInstance(expr.children[1], pybamm.MatrixMultiplication)

        # mat-mul on numerator and denominator
        expr = (m2 @ (m1 @ v) / (m2 @ (m1 @ v))).simplify()
        for child in expr.children:
            self.assertIsInstance(child.children[0], pybamm.Matrix)
            self.assertIsInstance(child.children[1], pybamm.StateVector)
            np.testing.assert_array_equal(
                child.children[0].entries, np.array([[6, 0], [0, 6]])
            )

        # mat-mul just on denominator
        expr = (1 / (m2 @ (m1 @ v))).simplify()
        self.assertIsInstance(expr.children[1].children[0], pybamm.Matrix)
        self.assertIsInstance(expr.children[1].children[1], pybamm.StateVector)
        np.testing.assert_array_equal(
            expr.children[1].children[0].entries, np.array([[6, 0], [0, 6]])
        )
        expr = (v2 / (m2 @ (m1 @ v))).simplify()
        self.assertIsInstance(expr.children[0], pybamm.StateVector)
        self.assertIsInstance(expr.children[1].children[0], pybamm.Matrix)
        self.assertIsInstance(expr.children[1].children[1], pybamm.StateVector)
        np.testing.assert_array_equal(
            expr.children[1].children[0].entries, np.array([[6, 0], [0, 6]])
        )

        # scalar * matrix
        for expr in [
            ((b * m1) @ v).simplify(),
            (b * (m1 @ v)).simplify(),
            ((m1 * b) @ v).simplify(),
            (m1 @ (b * v)).simplify(),
        ]:
            self.assertIsInstance(expr.children[0], pybamm.Matrix)
            self.assertIsInstance(expr.children[1], pybamm.StateVector)
            np.testing.assert_array_equal(
                expr.children[0].entries, np.array([[2, 0], [0, 2]])
            )

        # matrix * vector
        m1 = pybamm.Matrix(np.array([[2, 0], [0, 2]]))
        v1 = pybamm.Vector(np.array([1, 1]))

        for expr in [(m1 @ v1).simplify()]:
            self.assertIsInstance(expr, pybamm.Vector)
            np.testing.assert_array_equal(expr.entries, np.array([2, 2]))

        # dont expant mult within mat-mult (issue #253)
        m1 = pybamm.Matrix(np.ones((300, 299)))
        m2 = pybamm.Matrix(np.ones((299, 300)))
        m3 = pybamm.Matrix(np.ones((300, 300)))
        v1 = pybamm.StateVector(slice(0, 299))
        v2 = pybamm.StateVector(slice(0, 300))
        v3 = pybamm.Vector(np.ones(299))

        expr = m1 @ (v1 * m2)
        self.assertEqual(
            expr.simplify().evaluate(y=np.ones((299, 1))).shape, (300, 300)
        )
        np.testing.assert_array_equal(
            expr.evaluate(y=np.ones((299, 1))),
            expr.simplify().evaluate(y=np.ones((299, 1))),
        )

        # more complex expression
        expr2 = m1 @ (v1 * (m2 @ v2))
        expr2simp = expr2.simplify()
        np.testing.assert_array_equal(
            expr2.evaluate(y=np.ones(300)), expr2simp.evaluate(y=np.ones(300))
        )
        self.assertEqual(expr2.id, expr2simp.id)

        expr3 = m1 @ ((m2 @ v1) * (m3 @ v2))
        expr3simp = expr3.simplify()
        self.assertEqual(expr3.id, expr3simp.id)

        # more complex expression, with simplification
        expr3 = m1 @ (v3 * (m2 @ v2))
        expr3simp = expr3.simplify()
        self.assertNotEqual(expr3.id, expr3simp.id)
        np.testing.assert_array_equal(
            expr3.evaluate(y=np.ones(300)), expr3simp.evaluate(y=np.ones(300))
        )

        # we expect simplified solution to be much faster
        timer = pybamm.Timer()
        start = timer.time()
        for _ in range(100):
            expr3.evaluate(y=np.ones(300))
        end = timer.time()
        start_simp = timer.time()
        for _ in range(100):
            expr3simp.evaluate(y=np.ones(300))
        end_simp = timer.time()
        self.assertLess(end_simp - start_simp, 1.5 * (end - start))
        self.assertGreater(end - start, (end_simp - start_simp))

        m1 = pybamm.Matrix(np.ones((300, 300)))
        m2 = pybamm.Matrix(np.ones((300, 300)))
        m3 = pybamm.Matrix(np.ones((300, 300)))
        m4 = pybamm.Matrix(np.ones((300, 300)))
        v1 = pybamm.StateVector(slice(0, 300))
        v2 = pybamm.StateVector(slice(300, 600))
        v3 = pybamm.StateVector(slice(600, 900))
        v4 = pybamm.StateVector(slice(900, 1200))
        expr4 = (m1 @ v1) * ((m2 @ v2) / (m3 @ v3) - m4 @ v4)
        expr4simp = expr4.simplify()
        self.assertEqual(expr4.id, expr4simp.id)

        m2 = pybamm.Matrix(np.ones((299, 300)))
        v2 = pybamm.StateVector(slice(0, 300))
        v3 = pybamm.Vector(np.ones(299))
        exprs = [(m2 @ v2) * v3, (m2 @ v2) / v3]
        for expr in exprs:
            exprsimp = expr.simplify()
            self.assertIsInstance(exprsimp, pybamm.MatrixMultiplication)
            self.assertIsInstance(exprsimp.children[0], pybamm.Matrix)
            self.assertIsInstance(exprsimp.children[1], pybamm.StateVector)
            np.testing.assert_array_equal(
                expr.evaluate(y=np.ones(300)), exprsimp.evaluate(y=np.ones(300))
            )


    def test_matrix_modification(self):
        exp = self.mat @ self.mat + self.mat
        self.A[0, 0] = -1
        self.assertTrue(exp.children[1]._entries[0, 0], -1)
        self.assertTrue(exp.children[0].children[0]._entries[0, 0], -1)
        self.assertTrue(exp.children[0].children[1]._entries[0, 0], -1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
