#
# Test for the Simplify class
#
import math
import numpy as np
import pybamm
import unittest
from tests import get_discretisation_for_testing


class TestSimplify(unittest.TestCase):
    def test_symbol_simplify(self):
        a = pybamm.Scalar(0, domain="domain")
        b = pybamm.Scalar(1)
        c = pybamm.Parameter("c")
        d = pybamm.Scalar(-1)
        e = pybamm.Scalar(2)
        g = pybamm.Variable("g")
        gdot = pybamm.VariableDot("g'")

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

        def myfunction(x, y):
            return x * y

        f = pybamm.Function(myfunction, a, b)
        self.assertIsInstance((f).simplify(), pybamm.Scalar)
        self.assertEqual((f).simplify().evaluate(), 0)

        # FunctionParameter
        f = pybamm.FunctionParameter("function", {"b": b})
        self.assertIsInstance((f).simplify(), pybamm.FunctionParameter)
        self.assertEqual((f).simplify().children[0].id, b.id)

        f = pybamm.FunctionParameter("function", {"a": a, "b": b})
        self.assertIsInstance((f).simplify(), pybamm.FunctionParameter)
        self.assertEqual((f).simplify().children[0].id, a.id)
        self.assertEqual((f).simplify().children[1].id, b.id)

        # Gradient
        self.assertIsInstance((pybamm.grad(a)).simplify(), pybamm.Scalar)
        self.assertEqual((pybamm.grad(a)).simplify().evaluate(), 0)
        v = pybamm.Variable("v", domain="domain")
        grad_v = pybamm.grad(v)
        self.assertIsInstance(grad_v.simplify(), pybamm.Gradient)

        # Divergence
        div_b = pybamm.div(pybamm.PrimaryBroadcastToEdges(b, "domain"))
        self.assertIsInstance(div_b.simplify(), pybamm.PrimaryBroadcast)
        self.assertEqual(div_b.simplify().child.child.evaluate(), 0)
        self.assertIsInstance(
            (pybamm.div(pybamm.grad(v))).simplify(), pybamm.Divergence
        )

        # Integral
        self.assertIsInstance(
            (
                pybamm.Integral(a, pybamm.SpatialVariable("x", domain="domain"))
            ).simplify(),
            pybamm.Integral,
        )

        def_int = (pybamm.DefiniteIntegralVector(a, vector_type="column")).simplify()
        self.assertIsInstance(def_int, pybamm.DefiniteIntegralVector)
        self.assertEqual(def_int.vector_type, "column")

        bound_int = (pybamm.BoundaryIntegral(a, region="negative tab")).simplify()
        self.assertIsInstance(bound_int, pybamm.BoundaryIntegral)
        self.assertEqual(bound_int.region, "negative tab")

        # BoundaryValue
        v_neg = pybamm.Variable("v", domain=["negative electrode"])
        self.assertIsInstance(
            (pybamm.boundary_value(v_neg, "right")).simplify(), pybamm.BoundaryValue
        )

        # Delta function
        self.assertIsInstance(
            (pybamm.DeltaFunction(v_neg, "right", "domain")).simplify(),
            pybamm.DeltaFunction,
        )

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

        # addition and subtraction with matrix zero
        v = pybamm.Vector(np.zeros((10, 1)))
        self.assertIsInstance((b + v).simplify(), pybamm.Array)
        np.testing.assert_array_equal((b + v).simplify().evaluate(), np.ones((10, 1)))
        self.assertIsInstance((v + b).simplify(), pybamm.Array)
        np.testing.assert_array_equal((v + b).simplify().evaluate(), np.ones((10, 1)))
        self.assertIsInstance((b - v).simplify(), pybamm.Array)
        np.testing.assert_array_equal((b - v).simplify().evaluate(), np.ones((10, 1)))
        self.assertIsInstance((v - b).simplify(), pybamm.Array)
        np.testing.assert_array_equal((v - b).simplify().evaluate(), -np.ones((10, 1)))

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
        self.assertIsInstance(expr, pybamm.Addition)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 4.0)
        self.assertIsInstance(expr.children[1], pybamm.Negate)
        self.assertIsInstance(expr.children[1].children[0], pybamm.Parameter)

        expr = (e * g * b).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 2.0)
        self.assertIsInstance(expr.children[1], pybamm.Variable)

        expr = (e * gdot * b).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 2.0)
        self.assertIsInstance(expr.children[1], pybamm.VariableDot)

        expr = (e + (g - c)).simplify()
        self.assertIsInstance(expr, pybamm.Addition)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 2.0)
        self.assertIsInstance(expr.children[1], pybamm.Subtraction)
        self.assertIsInstance(expr.children[1].children[0], pybamm.Variable)
        self.assertIsInstance(expr.children[1].children[1], pybamm.Parameter)

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

        # not implemented for Symbol
        sym = pybamm.Symbol("sym")
        with self.assertRaises(NotImplementedError):
            sym.simplify()

        # A + A = 2A (#323)
        a = pybamm.Parameter("A")
        expr = (a + a).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 2)
        self.assertIsInstance(expr.children[1], pybamm.Parameter)

        expr = (a + a + a + a).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 4)
        self.assertIsInstance(expr.children[1], pybamm.Parameter)

        expr = (a - a + a - a + a + a).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 2)
        self.assertIsInstance(expr.children[1], pybamm.Parameter)

        # A - A = 0 (#323)
        expr = (a - a).simplify()
        self.assertIsInstance(expr, pybamm.Scalar)
        self.assertEqual(expr.evaluate(), 0)

        # B - (A+A) = B - 2*A (#323)
        expr = (b - (a + a)).simplify()
        self.assertIsInstance(expr, pybamm.Addition)
        self.assertIsInstance(expr.right, pybamm.Negate)
        self.assertIsInstance(expr.right.child, pybamm.Multiplication)
        self.assertEqual(expr.right.child.left.id, pybamm.Scalar(2).id)
        self.assertEqual(expr.right.child.right.id, a.id)

        # B - (1*A + 2*A) = B - 3*A (#323)
        expr = (b - (1 * a + 2 * a)).simplify()
        self.assertIsInstance(expr, pybamm.Addition)
        self.assertIsInstance(expr.right, pybamm.Negate)
        self.assertIsInstance(expr.right.child, pybamm.Multiplication)
        self.assertEqual(expr.right.child.left.id, pybamm.Scalar(3).id)
        self.assertEqual(expr.right.child.right.id, a.id)

        # B - (A + C) = B - (A + C) (not B - (A - C))
        expr = (b - (a + c)).simplify()
        self.assertIsInstance(expr, pybamm.Addition)
        self.assertIsInstance(expr.right, pybamm.Subtraction)
        self.assertEqual(expr.right.left.id, (-a).id)
        self.assertEqual(expr.right.right.id, c.id)

    def test_vector_zero_simplify(self):
        a1 = pybamm.Scalar(0)
        v1 = pybamm.Vector(np.zeros(10))
        a2 = pybamm.Scalar(1)
        v2 = pybamm.Vector(np.ones(10))

        for expr in [a1 * v1, v1 * a1, a2 * v1, v1 * a2, a1 * v2, v2 * a1, v1 * v2]:
            self.assertIsInstance(expr.simplify(), pybamm.Vector)
            np.testing.assert_array_equal(expr.simplify().entries, np.zeros((10, 1)))

    def test_matrix_simplifications(self):
        a = pybamm.Matrix(np.zeros((2, 2)))
        b = pybamm.Matrix(np.ones((2, 2)))

        # matrix multiplication
        A = pybamm.Matrix(np.array([[1, 0], [0, 1]]))
        self.assertIsInstance((a @ A).simplify(), pybamm.Matrix)
        np.testing.assert_array_equal(
            (a @ A).simplify().evaluate().toarray(), np.zeros((2, 2))
        )
        self.assertIsInstance((A @ a).simplify(), pybamm.Matrix)
        np.testing.assert_array_equal(
            (A @ a).simplify().evaluate().toarray(), np.zeros((2, 2))
        )

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

        expr = ((v * v) / 2).simplify()
        self.assertIsInstance(expr, pybamm.Multiplication)
        self.assertIsInstance(expr.children[0], pybamm.Scalar)
        self.assertEqual(expr.children[0].evaluate(), 0.5)
        self.assertIsInstance(expr.children[1], pybamm.Multiplication)

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
        b = pybamm.Scalar(1)
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
            np.testing.assert_array_equal(expr.entries, np.array([[2], [2]]))

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

        expr3 = m1 @ ((m2 @ v2) * (m2 @ v2))
        expr3simp = expr3.simplify()
        self.assertEqual(expr3.id, expr3simp.id)

        # more complex expression, with simplification
        expr3 = m1 @ (v3 * (m2 @ v2))
        expr3simp = expr3.simplify()
        self.assertNotEqual(expr3.id, expr3simp.id)
        np.testing.assert_array_equal(
            expr3.evaluate(y=np.ones(300)), expr3simp.evaluate(y=np.ones(300))
        )

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
        self.assertIsInstance(expr, pybamm.Vector)
        self.assertEqual(expr.shape, a.shape)
        np.testing.assert_array_equal(expr.evaluate(), 0)

        # zero matrix
        m1 = pybamm.Matrix(np.zeros((300, 300)))
        for expr in [m1 * v1, v1 * m1]:
            expr_simp = expr.simplify()
            self.assertIsInstance(expr_simp, pybamm.Matrix)
            np.testing.assert_array_equal(
                expr_simp.evaluate(y=np.ones(300)).toarray(), m1.evaluate()
            )

        # adding zero
        m2 = pybamm.Matrix(np.random.rand(300, 300))
        for expr in [m1 + m2, m2 + m1]:
            expr_simp = expr.simplify()
            self.assertIsInstance(expr_simp, pybamm.Matrix)
            np.testing.assert_array_equal(
                expr_simp.evaluate(y=np.ones(300)), m2.evaluate()
            )

        # subtracting zero
        for expr in [m1 - m2, -m2 - m1]:
            expr_simp = expr.simplify()
            self.assertIsInstance(expr_simp, pybamm.Matrix)
            np.testing.assert_array_equal(
                expr_simp.evaluate(y=np.ones(300)), -m2.evaluate()
            )

    def test_matrix_divide_simplify(self):
        m = pybamm.Matrix(np.random.rand(30, 20))
        zero = pybamm.Scalar(0)

        expr1 = (zero / m).simplify()
        self.assertIsInstance(expr1, pybamm.Matrix)
        self.assertEqual(expr1.shape, m.shape)
        np.testing.assert_array_equal(expr1.evaluate().toarray(), np.zeros((30, 20)))

        expr2 = (m / zero).simplify()
        self.assertIsInstance(expr2, pybamm.Matrix)
        self.assertEqual(expr2.shape, m.shape)
        np.testing.assert_array_equal(expr2.evaluate(), np.inf)

        m = pybamm.Matrix(np.zeros((10, 10)))
        a = pybamm.Scalar(7)
        expr3 = (m / a).simplify()
        self.assertIsInstance(expr3, pybamm.Matrix)
        self.assertEqual(expr3.shape, m.shape)
        np.testing.assert_array_equal(expr3.evaluate().toarray(), np.zeros((10, 10)))

    def test_domain_concatenation_simplify(self):
        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        a_dom = ["negative electrode"]
        b_dom = ["positive electrode"]
        a = 2 * pybamm.Vector(np.ones_like(mesh[a_dom[0]].nodes), domain=a_dom)
        b = pybamm.Vector(np.ones_like(mesh[b_dom[0]].nodes), domain=b_dom)

        conc = pybamm.DomainConcatenation([a, b], mesh)
        conc_simp = conc.simplify()

        # should be simplified to a vector
        self.assertIsInstance(conc_simp, pybamm.Vector)
        np.testing.assert_array_equal(
            conc_simp.evaluate(),
            np.concatenate(
                [
                    np.full((mesh[a_dom[0]].npts, 1), 2),
                    np.full((mesh[b_dom[0]].npts, 1), 1),
                ]
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

        y = mesh.combine_submeshes(*conc.domain).nodes ** 2
        self.assertIsInstance(conc_simp, pybamm.StateVector)
        self.assertEqual(len(conc_simp.y_slices), 1)
        self.assertEqual(conc_simp.y_slices[0].start, 0)
        self.assertEqual(conc_simp.y_slices[0].stop, len(y))
        np.testing.assert_array_equal(conc_disc.evaluate(y=y), conc_simp.evaluate(y=y))

    def test_simplify_broadcast(self):
        v = pybamm.StateVector(slice(0, 1))
        broad = pybamm.PrimaryBroadcast(v, "test")
        broad_simp = broad.simplify()
        self.assertEqual(broad_simp.id, broad.id)

    def test_simplify_heaviside(self):
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        self.assertEqual((a < b).simplify().id, pybamm.Scalar(1).id)
        self.assertEqual((a >= b).simplify().id, pybamm.Scalar(0).id)

    def test_simplify_inner(self):
        a1 = pybamm.Scalar(0)
        M1 = pybamm.Matrix(np.zeros((10, 10)))
        v1 = pybamm.Vector(np.ones(10))
        a2 = pybamm.Scalar(1)
        M2 = pybamm.Matrix(np.ones((10, 10)))
        a3 = pybamm.Scalar(3)

        np.testing.assert_array_equal(
            pybamm.inner(a1, M2).simplify().evaluate().toarray(), M1.entries
        )
        self.assertEqual(pybamm.inner(a1, a2).simplify().evaluate(), 0)
        np.testing.assert_array_equal(
            pybamm.inner(M2, a1).simplify().evaluate().toarray(), M1.entries
        )
        self.assertEqual(pybamm.inner(a2, a1).simplify().evaluate(), 0)
        np.testing.assert_array_equal(
            pybamm.inner(M1, a3).simplify().evaluate().toarray(), M1.entries
        )
        np.testing.assert_array_equal(
            pybamm.inner(v1, a3).simplify().evaluate(), 3 * v1.entries
        )
        self.assertEqual(pybamm.inner(a2, a3).simplify().evaluate(), 3)
        self.assertEqual(pybamm.inner(a3, a2).simplify().evaluate(), 3)
        self.assertEqual(pybamm.inner(a3, a3).simplify().evaluate(), 9)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
