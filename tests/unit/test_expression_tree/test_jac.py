#
# Tests for the jacobian methods
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
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

        func = 7 * u - 9 * v
        jacobian = np.array([[7, 0, -9, 0], [0, 7, 0, -9]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        A = pybamm.Matrix(2 * eye(2))
        func = A @ u
        jacobian = np.array([[2, 0, 0, 0], [0, 2, 0, 0]])
        dfunc_dy = func.jac(y).simplify().evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
