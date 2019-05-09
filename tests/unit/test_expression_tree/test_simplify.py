#
# Test for the Symbol class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from tests import get_discretisation_for_testing

import unittest
import numpy as np
import math


def test_const_function():
    return 1


class TestSimplify(unittest.TestCase):
    def test_symbol_simplify(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)
        d = pybamm.Scalar(-1)
        e = pybamm.Scalar(2)

        # negate
        self.assertIsInstance((-a).simplify(), pybamm.Scalar)
        self.assertEqual((-a).simplify().evaluate(), 0)
        self.assertIsInstance((-b).simplify(), pybamm.Scalar)
        self.assertEqual((-b).simplify().evaluate(), -1)

        # absolute value
        self.assertIsInstance((abs(a)).simplify(), pybamm.Scalar)
        self.assertEqual((abs(a)).simplify().evaluate(), 0)
        self.assertIsInstance((abs(d)).simplify(), pybamm.Scalar)
        self.assertEqual((abs(d)).simplify().evaluate(), 1)

        # function
        def sin(x):
            return math.sin(x)

        f = pybamm.Function(sin, b)
        self.assertIsInstance((f).simplify(), pybamm.Scalar)
        self.assertEqual((f).simplify().evaluate(), math.sin(1))

        # Gradient
        self.assertIsInstance((pybamm.grad(a)).simplify(), pybamm.Scalar)
        self.assertEqual((pybamm.grad(a)).simplify().evaluate(), 0)
        v = pybamm.Variable("v")
        self.assertIsInstance((pybamm.grad(v)).simplify(), pybamm.Gradient)

        # Divergence
        self.assertIsInstance((pybamm.div(a)).simplify(), pybamm.Scalar)
        self.assertEqual((pybamm.div(a)).simplify().evaluate(), 0)
        self.assertIsInstance((pybamm.div(v)).simplify(), pybamm.Divergence)

        # Integral
        self.assertIsInstance(
            (pybamm.Integral(a, pybamm.t)).simplify(), pybamm.Integral
        )

        # BoundaryValue
        v_neg = pybamm.Variable("v", domain=["negative electrode"])
        self.assertIsInstance((pybamm.surf(v_neg)).simplify(), pybamm.BoundaryValue)

        # addition
        self.assertIsInstance((a + b).simplify(), pybamm.Scalar)
        self.assertEqual((a + b).simplify().evaluate(), 1)
        self.assertIsInstance((b + b).simplify(), pybamm.Scalar)
        self.assertEqual((b + b).simplify().evaluate(), 2)
        self.assertIsInstance((b + a).simplify(), pybamm.Scalar)
        self.assertEqual((b + a).simplify().evaluate(), 1)

        # subtraction
        self.assertIsInstance((a - b).simplify(), pybamm.Scalar)
        self.assertEqual((a - b).simplify().evaluate(), -1)
        self.assertIsInstance((b - b).simplify(), pybamm.Scalar)
        self.assertEqual((b - b).simplify().evaluate(), 0)
        self.assertIsInstance((b - a).simplify(), pybamm.Scalar)
        self.assertEqual((b - a).simplify().evaluate(), 1)

        # multiplication
        self.assertIsInstance((a * b).simplify(), pybamm.Scalar)
        self.assertEqual((a * b).simplify().evaluate(), 0)
        self.assertIsInstance((b * a).simplify(), pybamm.Scalar)
        self.assertEqual((b * a).simplify().evaluate(), 0)
        self.assertIsInstance((b * b).simplify(), pybamm.Scalar)
        self.assertEqual((b * b).simplify().evaluate(), 1)
        self.assertIsInstance((a * a).simplify(), pybamm.Scalar)
        self.assertEqual((a * a).simplify().evaluate(), 0)

        # test when other node is a parameter
        c = pybamm.Parameter("c")
        self.assertIsInstance((a + c).simplify(), pybamm.Parameter)
        self.assertIsInstance((c + a).simplify(), pybamm.Parameter)
        self.assertIsInstance((c + b).simplify(), pybamm.Addition)
        self.assertIsInstance((b + c).simplify(), pybamm.Addition)
        self.assertIsInstance((a * c).simplify(), pybamm.Scalar)
        self.assertEqual((a * c).simplify().evaluate(), 0)
        self.assertIsInstance((c * a).simplify(), pybamm.Scalar)
        self.assertEqual((c * a).simplify().evaluate(), 0)
        self.assertIsInstance((b * c).simplify(), pybamm.Parameter)
        self.assertIsInstance((e * c).simplify(), pybamm.Multiplication)

        expr = (e * (e * c)).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertIsInstance(expr.children[1], pybamm.Parameter)

        expr = (e / (e * c)).simplify()
        self.assertIsInstance(expr, pybamm.Division)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 1.0)
        self.assertIsInstance(expr.children[1], pybamm.Parameter)

        expr = (e * (e / c)).simplify()
        self.assertIsInstance(expr, pybamm.Division)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 4.0)
        self.assertIsInstance(expr.children[1], pybamm.Parameter)

        expr = (e * (c / e)).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 1.0)
        self.assertIsInstance(expr.children[1], pybamm.Parameter)

        expr = ((e * c) * (c / e)).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 1.0)
        self.assertIsInstance(expr.children[1], pybamm.Multiplication)
        self.assertIsInstance(expr.children[1].children[0], pybamm.Parameter)
        self.assertIsInstance(expr.children[1].children[1], pybamm.Parameter)

        expr = (e + (e + c)).simplify()
        self.assertIsInstance(expr, pybamm.Addition)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 4.0)
        self.assertIsInstance(expr.children[1], pybamm.Parameter)

        expr = (e + (e - c)).simplify()
        self.assertIsInstance(expr, pybamm.Subtraction)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 4.0)
        self.assertIsInstance(expr.children[1], pybamm.Parameter)

        expr = ((2 + c) + (c + 2)).simplify()
        self.assertIsInstance(expr, pybamm.Addition)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 4.0)
        self.assertIsInstance(expr.children[1], pybamm.Multiplication)
        self.assertIsInstance(expr.children[1].children[0], pybamm.Scalar)
        self.assertEqual(expr.children[1].children[0].evaluate(), 2)
        self.assertIsInstance(expr.children[1].children[1], pybamm.Parameter)

        expr = ((-1 + c) - (c + 1) + (c - 1)).simplify()
        self.assertIsInstance(expr, pybamm.Addition)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), -3.0)

        # check these don't simplify
        self.assertIsInstance((c * e).simplify(), pybamm.Multiplication)
        self.assertIsInstance((e / c).simplify(), pybamm.Division)
        self.assertIsInstance((c).simplify(), pybamm.Parameter)
        c1 = pybamm.Parameter("c1")
        self.assertIsInstance((c1 * c).simplify(), pybamm.Multiplication)

        # should simplify division to multiply
        self.assertIsInstance((c / e).simplify(), pybamm.Multiplication)

        self.assertIsInstance((c / b).simplify(), pybamm.Parameter)
        self.assertIsInstance((c * b).simplify(), pybamm.Parameter)

        # negation with parameter
        self.assertIsInstance((-c).simplify(), pybamm.Negate)

        self.assertIsInstance((a + b + a).simplify(), pybamm.Scalar)
        self.assertEqual((a + b + a).simplify().evaluate(), 1)
        self.assertIsInstance((b + a + a).simplify(), pybamm.Scalar)
        self.assertEqual((b + a + a).simplify().evaluate(), 1)
        self.assertIsInstance((a * b * b).simplify(), pybamm.Scalar)
        self.assertEqual((a * b * b).simplify().evaluate(), 0)
        self.assertIsInstance((b * a * b).simplify(), pybamm.Scalar)
        self.assertEqual((b * a * b).simplify().evaluate(), 0)

        # power simplification
        self.assertIsInstance((c ** a).simplify(), pybamm.Scalar)
        self.assertEqual((c ** a).simplify().evaluate(), 1)
        self.assertIsInstance((a ** c).simplify(), pybamm.Scalar)
        self.assertEqual((a ** c).simplify().evaluate(), 0)
        d = pybamm.Scalar(2)
        self.assertIsInstance((c ** d).simplify(), pybamm.Power)

        # division
        self.assertIsInstance((a / b).simplify(), pybamm.Scalar)
        self.assertEqual((a / b).simplify().evaluate(), 0)
        self.assertIsInstance((b / a).simplify(), pybamm.Scalar)
        self.assertEqual((b / a).simplify().evaluate(), np.inf)
        self.assertIsInstance((a / a).simplify(), pybamm.Scalar)
        self.assertTrue(np.isnan((a / a).simplify().evaluate()))
        self.assertIsInstance((b / b).simplify(), pybamm.Scalar)
        self.assertEqual((b / b).simplify().evaluate(), 1)

        # Spatial variable
        x = pybamm.SpatialVariable("x", ["negative electrode"])
        self.assertIsInstance(x.simplify(), pybamm.SpatialVariable)
        self.assertEqual(x.simplify().id, x.id)

        # not implemented for Symbol
        sym = pybamm.Symbol("sym")
        with self.assertRaises(NotImplementedError):
            sym.simplify()

        # A + A = 2A (#323)
        a = pybamm.Variable('A')
        expr = (a + a).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 2)
        self.assertIsInstance(expr.children[1], pybamm.Variable)

        expr = (a + a + a + a).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 4)
        self.assertIsInstance(expr.children[1], pybamm.Variable)

        expr = (a - a + a - a + a + a).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 2)
        self.assertIsInstance(expr.children[1], pybamm.Variable)

        # A - A = 0 (#323)
        expr = (a - a).simplify()
        self.assertIsInstance(expr, pybamm.Scalar)
        self.assertEqual(expr.evaluate(), 0)

    def test_function_simplify(self):
        a = pybamm.Parameter("a")
        funca = pybamm.Function(test_const_function, a).simplify()
        self.assertIsInstance(funca, pybamm.Scalar)
        self.assertEqual(funca.evaluate(), 1)

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
        for _ in range(200):
            expr3.evaluate(y=np.ones(300))
        end = timer.time()
        start_simp = timer.time()
        for _ in range(200):
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

        # A + A = 2A (#323)
        a = pybamm.StateVector(slice(0, 300))
        expr = (a + a).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 2)
        self.assertIsInstance(expr.children[1], pybamm.StateVector)

        expr = (a + a + a + a).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 4)
        self.assertIsInstance(expr.children[1], pybamm.StateVector)

        # A - A = 0 (#323)
        expr = (a - a).simplify()
        self.assertIsInstance(expr, pybamm.Scalar)
        self.assertEqual(expr.evaluate(), 0)

    def test_domain_concatenation_simplify(self):
        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        a_dom = ["negative electrode"]
        b_dom = ["positive electrode"]
        a = pybamm.NumpyBroadcast(pybamm.Scalar(2), a_dom, mesh)
        b = pybamm.Vector(np.ones_like(mesh[b_dom[0]][0].nodes), domain=b_dom)

        conc = pybamm.DomainConcatenation([a, b], mesh)
        conc_simp = conc.simplify()

        # should be simplified to a vector
        self.assertIsInstance(conc_simp, pybamm.Vector)
        np.testing.assert_array_equal(
            conc_simp.evaluate(),
            np.concatenate(
                [np.full(mesh[a_dom[0]][0].npts, 2), np.full(mesh[b_dom[0]][0].npts, 1)]
            ),
        )

    def test_simplify_concatenation_state_vectors(self):
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        a = pybamm.Variable("a", domain=["negative electrode"])
        b = pybamm.Variable("b", domain=["separator"])
        c = pybamm.Variable("c", domain=["positive electrode"])
        conc = pybamm.Concatenation(a, b, c)
        disc.set_variable_slices([a, b, c])
        conc_disc = disc.process_symbol(conc)
        conc_simp = conc_disc.simplify()

        y = mesh.combine_submeshes(*conc.domain)[0].nodes ** 2
        self.assertIsInstance(conc_simp, pybamm.StateVector)
        self.assertEqual(conc_simp.y_slice.start, 0)
        self.assertEqual(conc_simp.y_slice.stop, len(y))
        np.testing.assert_array_equal(conc_disc.evaluate(y=y), conc_simp.evaluate(y=y))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
