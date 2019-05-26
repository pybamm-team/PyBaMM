#
# Tests for the jacobian methods
#
import pybamm

import numpy as np
import autograd.numpy as auto_np
import unittest
from scipy.sparse import eye


class TestJacobian(unittest.TestCase):
    def test_linear(self):
        y = pybamm.StateVector(slice(0, 4))
        u = pybamm.StateVector(slice(0, 2))
        v = pybamm.StateVector(slice(2, 4))

        y0 = np.ones(4)

        func = u
        jacobian = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = -v
        jacobian = np.array([[0, 0, -1, 0], [0, 0, 0, -1]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = 3 * u + 4 * v
        jacobian = np.array([[3, 0, 4, 0], [0, 3, 0, 4]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = 7 * u - v * 9
        jacobian = np.array([[7, 0, -9, 0], [0, 7, 0, -9]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        A = pybamm.Matrix(2 * eye(2))
        func = A @ u
        jacobian = np.array([[2, 0, 0, 0], [0, 2, 0, 0]])
        dfunc_dy = func.jac(y).simplify().evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = u @ A
        with self.assertRaises(NotImplementedError):
            func.jac(y)

    def test_nonlinear(self):
        y = pybamm.StateVector(slice(0, 4))
        u = pybamm.StateVector(slice(0, 2))
        v = pybamm.StateVector(slice(2, 4))

        y0 = np.array([1, 2, 3, 4])

        func = v ** 2
        jacobian = np.array([[0, 0, 6, 0], [0, 0, 0, 8]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = 2 ** v
        jacobian = np.array(
            [[0, 0, 2 ** 3 * np.log(2), 0], [0, 0, 0, 2 ** 4 * np.log(2)]]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = v ** v
        jacobian = [[0, 0, 27 * (1 + np.log(3)), 0], [0, 0, 0, 256 * (1 + np.log(4))]]
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_almost_equal(jacobian, dfunc_dy.toarray())

        func = u * v
        jacobian = np.array([[3, 0, 1, 0], [0, 4, 0, 2]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = u * (u + v)
        jacobian = np.array([[5, 0, 1, 0], [0, 8, 0, 2]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = 1 / u + v / 3
        jacobian = np.array([[-1, 0, 1 / 3, 0], [0, -1 / 4, 0, 1 / 3]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = u / v
        jacobian = np.array([[1 / 3, 0, -1 / 9, 0], [0, 1 / 4, 0, -1 / 8]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = v / (1 + v)
        jacobian = np.array([[0, 0, 1 / 16, 0], [0, 0, 0, 1 / 25]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = pybamm.AbsoluteValue(v)
        with self.assertRaises(pybamm.UndefinedOperation):
            func.jac(y)

    def test_functions(self):
        y = pybamm.StateVector(slice(0, 4))
        u = pybamm.StateVector(slice(0, 2))
        v = pybamm.StateVector(slice(2, 4))

        y0 = np.array([1.0, 2.0, 3.0, 4.0])

        func = pybamm.Function(auto_np.sin, u)
        jacobian = np.array([[np.cos(1), 0, 0, 0], [0, np.cos(2), 0, 0]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = pybamm.Function(auto_np.cos, v)
        jacobian = np.array([[0, 0, -np.sin(3), 0], [0, 0, 0, -np.sin(4)]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = pybamm.Function(auto_np.sin, 3 * u * v)
        jacobian = np.array(
            [
                [9 * np.cos(9), 0, 3 * np.cos(9), 0],
                [0, 12 * np.cos(24), 0, 6 * np.cos(24)],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = pybamm.Function(auto_np.cos, 5 * pybamm.Function(auto_np.exp, u + v))
        jacobian = np.array(
            [
                [
                    -5 * np.exp(4) * np.sin(5 * np.exp(4)),
                    0,
                    -5 * np.exp(4) * np.sin(5 * np.exp(4)),
                    0,
                ],
                [
                    0,
                    -5 * np.exp(6) * np.sin(5 * np.exp(6)),
                    0,
                    -5 * np.exp(6) * np.sin(5 * np.exp(6)),
                ],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

    def test_index(self):
        vec = pybamm.StateVector(slice(0, 5))
        ind = pybamm.Index(vec, 3)
        jac = ind.jac(vec).evaluate(y=np.linspace(0, 2, 5)).toarray()
        np.testing.assert_array_equal(jac, np.array([[0, 0, 0, 1, 0]]))

    def test_jac_of_self(self):
        "Jacobian of variable with respect to itself should be one."
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")

        self.assertEqual(a.jac(a).evaluate(), 1)

        add = a + b
        self.assertEqual(add.jac(add).evaluate(), 1)

        subtract = a - b
        self.assertEqual(subtract.jac(subtract).evaluate(), 1)

        multiply = a * b
        self.assertEqual(multiply.jac(multiply).evaluate(), 1)

        divide = a / b
        self.assertEqual(divide.jac(divide).evaluate(), 1)

        power = a ** b
        self.assertEqual(power.jac(power).evaluate(), 1)

    def test_jac_of_number(self):
        "Jacobian of a number should be zero"
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)

        y = pybamm.Variable("y")

        self.assertEqual(a.jac(y).evaluate(), 0)

        add = a + b
        self.assertEqual(add.jac(y).evaluate(), 0)

        subtract = a - b
        self.assertEqual(subtract.jac(y).evaluate(), 0)

        multiply = a * b
        self.assertEqual(multiply.jac(y).evaluate(), 0)

        divide = a / b
        self.assertEqual(divide.jac(y).evaluate(), 0)

        power = a ** b
        self.assertEqual(power.jac(y).evaluate(), 0)

    def test_jac_of_symbol(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")

        self.assertEqual(a.jac(b).evaluate(), 0)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
