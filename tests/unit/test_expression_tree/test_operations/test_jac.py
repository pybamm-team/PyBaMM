#
# Tests for the jacobian methods
#

import pybamm

import numpy as np
import unittest
from scipy.sparse import eye
from tests import get_mesh_for_testing


class TestJacobian(unittest.TestCase):
    def test_variable_is_statevector(self):
        a = pybamm.Symbol("a")
        with self.assertRaisesRegex(
            TypeError, "Jacobian can only be taken with respect to a 'StateVector'"
        ):
            a.jac(a)

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
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = u @ pybamm.StateVector(slice(0, 1))
        with self.assertRaises(NotImplementedError):
            func.jac(y)

        # when differentiating by independent part of the state vector
        jacobian = np.array([[0, 0], [0, 0]])
        du_dv = u.jac(v).evaluate().toarray()
        np.testing.assert_array_equal(du_dv, jacobian)

    def test_nonlinear(self):
        y = pybamm.StateVector(slice(0, 4))
        u = pybamm.StateVector(slice(0, 2))
        v = pybamm.StateVector(slice(2, 4))

        y0 = np.array([1, 2, 3, 4])

        func = v**2
        jacobian = np.array([[0, 0, 6, 0], [0, 0, 0, 8]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = 2**v
        jacobian = np.array([[0, 0, 2**3 * np.log(2), 0], [0, 0, 0, 2**4 * np.log(2)]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = v**v
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

    def test_multislice_raises(self):
        y1 = pybamm.StateVector(slice(0, 4), slice(7, 8))
        y_dot1 = pybamm.StateVectorDot(slice(0, 4), slice(7, 8))
        y2 = pybamm.StateVector(slice(4, 7))
        with self.assertRaises(NotImplementedError):
            y1.jac(y1)
        with self.assertRaises(NotImplementedError):
            y2.jac(y1)
        with self.assertRaises(NotImplementedError):
            y_dot1.jac(y1)

    def test_linear_ydot(self):
        y = pybamm.StateVector(slice(0, 4))
        y_dot = pybamm.StateVectorDot(slice(0, 4))
        u = pybamm.StateVector(slice(0, 2))
        v = pybamm.StateVector(slice(2, 4))
        u_dot = pybamm.StateVectorDot(slice(0, 2))
        v_dot = pybamm.StateVectorDot(slice(2, 4))

        y0 = np.ones(4)
        y_dot0 = np.ones(4)

        func = u_dot
        jacobian = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        dfunc_dy = func.jac(y_dot).evaluate(y=y0, y_dot=y_dot0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = -v_dot
        jacobian = np.array([[0, 0, -1, 0], [0, 0, 0, -1]])
        dfunc_dy = func.jac(y_dot).evaluate(y=y0, y_dot=y_dot0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = u_dot
        jacobian = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        dfunc_dy = func.jac(y).evaluate(y=y0, y_dot=y_dot0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = -v_dot
        jacobian = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        dfunc_dy = func.jac(y).evaluate(y=y0, y_dot=y_dot0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = u
        jacobian = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        dfunc_dy = func.jac(y_dot).evaluate(y=y0, y_dot=y_dot0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = -v
        jacobian = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        dfunc_dy = func.jac(y_dot).evaluate(y=y0, y_dot=y_dot0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

    def test_functions(self):
        y = pybamm.StateVector(slice(0, 4))
        u = pybamm.StateVector(slice(0, 2))
        v = pybamm.StateVector(slice(2, 4))
        const = pybamm.Scalar(1)

        y0 = np.array([1.0, 2.0, 3.0, 4.0])

        func = pybamm.sin(u)
        jacobian = np.array([[np.cos(1), 0, 0, 0], [0, np.cos(2), 0, 0]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = pybamm.cos(v)
        jacobian = np.array([[0, 0, -np.sin(3), 0], [0, 0, 0, -np.sin(4)]])
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = pybamm.sin(3 * u * v)
        jacobian = np.array(
            [
                [9 * np.cos(9), 0, 3 * np.cos(9), 0],
                [0, 12 * np.cos(24), 0, 6 * np.cos(24)],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = pybamm.cos(5 * pybamm.exp(u + v))
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

        # when child evaluates to number
        func = pybamm.Sin(const)
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(0, dfunc_dy)

    def test_index(self):
        vec = pybamm.StateVector(slice(0, 5))
        ind = pybamm.Index(vec, 3)
        jac = ind.jac(vec).evaluate(y=np.linspace(0, 2, 5)).toarray()
        np.testing.assert_array_equal(jac, np.array([[0, 0, 0, 1, 0]]))

        # jac of ind of something that isn't a StateVector should return zeros
        const_vec = pybamm.Vector(np.ones(3))
        ind = pybamm.Index(const_vec, 2)
        jac = ind.jac(vec).evaluate(y=np.linspace(0, 2, 5)).toarray()
        np.testing.assert_array_equal(jac, np.array([[0, 0, 0, 0, 0]]))

    def test_evaluate_at(self):
        y = pybamm.StateVector(slice(0, 4))
        expr = pybamm.EvaluateAt(y, 2)
        jac = expr.jac(y).evaluate(y=np.linspace(0, 2, 4))
        np.testing.assert_array_equal(jac, 0)

    def test_jac_of_number(self):
        """Jacobian of a number should be zero"""
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)

        y = pybamm.StateVector(slice(0, 1))

        self.assertEqual(a.jac(y).evaluate(), 0)

        add = a + b
        self.assertEqual(add.jac(y).evaluate(), 0)

        subtract = a - b
        self.assertEqual(subtract.jac(y).evaluate(), 0)

        multiply = a * b
        self.assertEqual(multiply.jac(y).evaluate(), 0)

        divide = a / b
        self.assertEqual(divide.jac(y).evaluate(), 0)

        power = a**b
        self.assertEqual(power.jac(y).evaluate(), 0)

    def test_jac_of_symbol(self):
        a = pybamm.Symbol("a")
        y = pybamm.StateVector(slice(0, 1))
        with self.assertRaises(NotImplementedError):
            a.jac(y)

    def test_spatial_operator(self):
        a = pybamm.Variable("a")
        b = pybamm.SpatialOperator("Operator", a)
        y = pybamm.StateVector(slice(0, 1))
        with self.assertRaises(NotImplementedError):
            b.jac(y)

    def test_jac_of_unary_operator(self):
        a = pybamm.Scalar(1)
        b = pybamm.UnaryOperator("Operator", a)
        y = pybamm.StateVector(slice(0, 1))
        with self.assertRaises(NotImplementedError):
            b.jac(y)

    def test_jac_of_binary_operator(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")

        phi_s = pybamm.Variable(
            "Negative electrode potential [V]", domain="negative electrode"
        )
        i = pybamm.grad(phi_s)

        inner = pybamm.inner(2, i)
        self.assertEqual(inner._binary_jac(a, b), 2 * b)

        inner = pybamm.inner(i, 2)
        self.assertEqual(inner._binary_jac(a, b), 2 * a)

        inner = pybamm.inner(i, i)
        self.assertEqual(inner._binary_jac(a, b), i * a + i * b)

    def test_jac_of_independent_variable(self):
        a = pybamm.IndependentVariable("Variable")
        y = pybamm.StateVector(slice(0, 1))
        self.assertEqual(a.jac(y).evaluate(), 0)

    def test_jac_of_inner(self):
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        y = pybamm.StateVector(slice(0, 1))
        self.assertEqual(pybamm.inner(a, b).jac(y).evaluate(), 0)
        self.assertEqual(pybamm.inner(a, y).jac(y).evaluate(), 1)
        self.assertEqual(pybamm.inner(y, b).jac(y).evaluate(), 2)
        vec = pybamm.StateVector(slice(0, 2))
        jac = pybamm.inner(a * vec, b * vec).jac(vec).evaluate(y=np.ones(2)).toarray()
        np.testing.assert_array_equal(jac, 4 * np.eye(2))

    def test_jac_of_heaviside(self):
        a = pybamm.Scalar(1)
        y = pybamm.StateVector(slice(0, 5))
        np.testing.assert_array_equal(
            ((a < y) * y**2).jac(y).evaluate(y=5 * np.ones(5)).toarray(),
            10 * np.eye(5),
        )
        np.testing.assert_array_equal(
            ((a < y) * y**2).jac(y).evaluate(y=-5 * np.ones(5)).toarray(), 0
        )

    def test_jac_of_equality(self):
        a = pybamm.Scalar(1)
        y = pybamm.StateVector(slice(0, 1))
        np.testing.assert_array_equal(
            (pybamm.Equality(a, y)).jac(y).evaluate(),
            0,
        )

    def test_jac_of_modulo(self):
        a = pybamm.Scalar(3)
        y = pybamm.StateVector(slice(0, 5))
        np.testing.assert_array_equal(
            (a % (3 * a)).jac(y).evaluate(y=5 * np.ones(5)), 0
        )
        np.testing.assert_array_equal(
            ((y % a) * y**2).jac(y).evaluate(y=5 * np.ones(5)).toarray(),
            45 * np.eye(5),
        )
        np.testing.assert_array_equal(
            ((a % y) * y**2).jac(y).evaluate(y=5 * np.ones(5)).toarray(),
            30 * np.eye(5),
        )
        np.testing.assert_array_equal(
            (((y + 1) ** 2 % y) * y**2).jac(y).evaluate(y=5 * np.ones(5)).toarray(),
            135 * np.eye(5),
        )

    def test_jac_of_minimum_maximum(self):
        y = pybamm.StateVector(slice(0, 10))
        y_test = np.linspace(0, 2, 10)
        np.testing.assert_array_equal(
            np.diag(pybamm.minimum(1, y**2).jac(y).evaluate(y=y_test).toarray()),
            2 * y_test * (y_test < 1),
        )
        np.testing.assert_array_equal(
            np.diag(pybamm.maximum(1, y**2).jac(y).evaluate(y=y_test).toarray()),
            2 * y_test * (y_test > 1),
        )

    def test_jac_of_abs(self):
        y = pybamm.StateVector(slice(0, 10))
        absy = abs(y)
        jac = absy.jac(y)
        y_test = np.linspace(-2, 2, 10)
        np.testing.assert_array_equal(
            np.diag(jac.evaluate(y=y_test).toarray()), np.sign(y_test)
        )

    def test_jac_of_sign(self):
        y = pybamm.StateVector(slice(0, 10))
        func = pybamm.sign(y) * y
        jac = func.jac(y)
        y_test = np.linspace(-2, 2, 10)
        np.testing.assert_array_equal(
            np.diag(jac.evaluate(y=y_test).toarray()), np.sign(y_test)
        )

    def test_jac_of_floor(self):
        y = pybamm.StateVector(slice(0, 10))
        func = pybamm.Floor(y) * y
        jac = func.jac(y)
        y_test = np.linspace(-2, 2, 10)
        np.testing.assert_array_equal(
            np.diag(jac.evaluate(y=y_test).toarray()), np.floor(y_test)
        )

    def test_jac_of_ceiling(self):
        y = pybamm.StateVector(slice(0, 10))
        func = pybamm.Ceiling(y) * y
        jac = func.jac(y)
        y_test = np.linspace(-2, 2, 10)
        np.testing.assert_array_equal(
            np.diag(jac.evaluate(y=y_test).toarray()), np.ceil(y_test)
        )

    def test_jac_of_numpy_concatenation(self):
        u = pybamm.StateVector(slice(0, 2))

        y0 = np.ones(2)

        # Multiple children
        func = pybamm.NumpyConcatenation(u, u)
        jacobian = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        dfunc_dy = func.jac(u).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        # One child
        self.assertEqual(u.jac(u), pybamm.NumpyConcatenation(u).jac(u))

    def test_jac_of_domain_concatenation(self):
        # create mesh
        mesh = get_mesh_for_testing()
        y = pybamm.StateVector(slice(0, 100))

        # Jacobian of a DomainConcatenation of constants is a zero matrix of the
        # appropriate size
        a_dom = ["negative electrode"]
        b_dom = ["separator"]
        c_dom = ["positive electrode"]
        a_npts = mesh[a_dom[0]].npts
        b_npts = mesh[b_dom[0]].npts
        c_npts = mesh[c_dom[0]].npts
        a = 2 * pybamm.Vector(np.ones(a_npts), domain=a_dom)
        b = pybamm.Vector(np.ones(b_npts), domain=b_dom)
        c = 3 * pybamm.Vector(np.ones(c_npts), domain=c_dom)

        conc = pybamm.DomainConcatenation([a, b, c], mesh)
        jac = conc.jac(y).evaluate().toarray()
        np.testing.assert_array_equal(jac, np.zeros((100, 100)))

        # Jacobian of a DomainConcatenation of StateVectors
        a = 2 * pybamm.StateVector(slice(0, a_npts), domain=a_dom)
        b = pybamm.StateVector(slice(a_npts, a_npts + b_npts), domain=b_dom)
        c = 3 * pybamm.StateVector(
            slice(a_npts + b_npts, a_npts + b_npts + c_npts), domain=c_dom
        )
        conc = pybamm.DomainConcatenation([a, b, c], mesh)

        y0 = np.ones(100)
        jac = conc.jac(y).evaluate(y=y0).toarray()
        np.testing.assert_array_equal(
            jac,
            np.diag(
                np.concatenate(
                    [2 * np.ones(a_npts), np.ones(b_npts), 3 * np.ones(c_npts)]
                )
            ),
        )

        # multi=domain case not implemented
        a = 2 * pybamm.StateVector(slice(0, a_npts), domain=a_dom)
        b = pybamm.StateVector(
            slice(a_npts, a_npts + b_npts + c_npts), domain=b_dom + c_dom
        )
        conc = pybamm.DomainConcatenation([a, b], mesh)
        with self.assertRaisesRegex(
            NotImplementedError, "jacobian only implemented for when each child has"
        ):
            conc.jac(y)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
