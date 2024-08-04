#
# Tests for the jacobian methods for two-dimensional objects
#

import pybamm

import numpy as np
import unittest
from scipy.sparse import eye
from tests import (
    get_1p1d_discretisation_for_testing,
)


class TestJacobian(unittest.TestCase):
    def test_linear(self):
        y = pybamm.StateVector(slice(0, 8))
        u = pybamm.StateVector(slice(0, 2), slice(4, 6))
        v = pybamm.StateVector(slice(2, 4), slice(6, 8))

        y0 = np.ones(8)

        func = u
        jacobian = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = -v
        jacobian = np.array(
            [
                [0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, -1],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = 3 * u + 4 * v
        jacobian = np.array(
            [
                [3, 0, 4, 0, 0, 0, 0, 0],
                [0, 3, 0, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 3, 0, 4, 0],
                [0, 0, 0, 0, 0, 3, 0, 4],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = 7 * u - v * 9
        jacobian = np.array(
            [
                [7, 0, -9, 0, 0, 0, 0, 0],
                [0, 7, 0, -9, 0, 0, 0, 0],
                [0, 0, 0, 0, 7, 0, -9, 0],
                [0, 0, 0, 0, 0, 7, 0, -9],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        A = pybamm.Matrix(2 * eye(4))
        func = A @ u
        jacobian = np.array(
            [
                [2, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 0, 0],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        # when differentiating by independent part of the state vector
        with self.assertRaises(NotImplementedError):
            u.jac(v)

    def test_nonlinear(self):
        y = pybamm.StateVector(slice(0, 8))
        u = pybamm.StateVector(slice(0, 2), slice(4, 6))
        v = pybamm.StateVector(slice(2, 4), slice(6, 8))

        y0 = np.arange(1, 9)

        func = v**2
        jacobian = np.array(
            [
                [0, 0, 6, 0, 0, 0, 0, 0],
                [0, 0, 0, 8, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 14, 0],
                [0, 0, 0, 0, 0, 0, 0, 16],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = 2**v
        jacobian = np.array(
            [
                [0, 0, 2**3 * np.log(2), 0, 0, 0, 0, 0],
                [0, 0, 0, 2**4 * np.log(2), 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2**7 * np.log(2), 0],
                [0, 0, 0, 0, 0, 0, 0, 2**8 * np.log(2)],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = v**v
        jacobian = [
            [0, 0, 27 * (1 + np.log(3)), 0, 0, 0, 0, 0],
            [0, 0, 0, 256 * (1 + np.log(4)), 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 823543 * (1 + np.log(7)), 0],
            [0, 0, 0, 0, 0, 0, 0, 16777216 * (1 + np.log(8))],
        ]
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_almost_equal(jacobian, dfunc_dy.toarray())

        func = u * v
        jacobian = np.array(
            [
                [3, 0, 1, 0, 0, 0, 0, 0],
                [0, 4, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 7, 0, 5, 0],
                [0, 0, 0, 0, 0, 8, 0, 6],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = u * (u + v)
        jacobian = np.array(
            [
                [5, 0, 1, 0, 0, 0, 0, 0],
                [0, 8, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 17, 0, 5, 0],
                [0, 0, 0, 0, 0, 20, 0, 6],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

    def test_functions(self):
        y = pybamm.StateVector(slice(0, 8))
        u = pybamm.StateVector(slice(0, 2), slice(4, 6))
        v = pybamm.StateVector(slice(2, 4), slice(6, 8))

        y0 = np.arange(1, 9)
        const = pybamm.Scalar(1)

        func = pybamm.sin(u)
        jacobian = np.array(
            [
                [np.cos(1), 0, 0, 0, 0, 0, 0, 0],
                [0, np.cos(2), 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, np.cos(5), 0, 0, 0],
                [0, 0, 0, 0, 0, np.cos(6), 0, 0],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = pybamm.cos(v)
        jacobian = np.array(
            [
                [0, 0, -np.sin(3), 0, 0, 0, 0, 0],
                [0, 0, 0, -np.sin(4), 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, -np.sin(7), 0],
                [0, 0, 0, 0, 0, 0, 0, -np.sin(8)],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        func = pybamm.sin(3 * u * v)
        jacobian = np.array(
            [
                [9 * np.cos(9), 0, 3 * np.cos(9), 0, 0, 0, 0, 0],
                [0, 12 * np.cos(24), 0, 6 * np.cos(24), 0, 0, 0, 0],
                [0, 0, 0, 0, 21 * np.cos(105), 0, 15 * np.cos(105), 0],
                [0, 0, 0, 0, 0, 24 * np.cos(144), 0, 18 * np.cos(144)],
            ]
        )
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(jacobian, dfunc_dy.toarray())

        # when child evaluates to number
        func = pybamm.sin(const)
        dfunc_dy = func.jac(y).evaluate(y=y0)
        np.testing.assert_array_equal(0, dfunc_dy)

    def test_jac_of_domain_concatenation(self):
        # create mesh
        disc = get_1p1d_discretisation_for_testing()
        mesh = disc.mesh
        y = pybamm.StateVector(slice(0, 1500))

        # Jacobian of a DomainConcatenation of constants is a zero matrix of the
        # appropriate size
        a_dom = ["negative electrode"]
        b_dom = ["separator"]
        c_dom = ["positive electrode"]
        cc_npts = mesh["current collector"].npts
        curr_coll_vector = pybamm.Vector(np.ones(cc_npts), domain="current collector")
        a = 2 * pybamm.PrimaryBroadcast(curr_coll_vector, a_dom)
        b = pybamm.PrimaryBroadcast(curr_coll_vector, b_dom)
        c = 3 * pybamm.PrimaryBroadcast(curr_coll_vector, c_dom)
        # Add bounds for compatibility with the discretisation
        a.bounds = (pybamm.Scalar(-np.inf), pybamm.Scalar(np.inf))
        b.bounds = (pybamm.Scalar(-np.inf), pybamm.Scalar(np.inf))
        c.bounds = (pybamm.Scalar(-np.inf), pybamm.Scalar(np.inf))

        conc = pybamm.concatenation(a, b, c)
        conc.bounds = a.bounds
        disc.set_variable_slices([conc])
        conc_disc = disc.process_symbol(conc)
        jac = conc_disc.jac(y).evaluate().toarray()
        np.testing.assert_array_equal(jac, np.zeros((1500, 1500)))

        # Jacobian of a DomainConcatenation of StateVectors
        a = pybamm.Variable(
            "a", domain=a_dom, auxiliary_domains={"secondary": "current collector"}
        )
        b = pybamm.Variable(
            "b", domain=b_dom, auxiliary_domains={"secondary": "current collector"}
        )
        c = pybamm.Variable(
            "c", domain=c_dom, auxiliary_domains={"secondary": "current collector"}
        )
        conc = pybamm.concatenation(a, b, c)
        disc.set_variable_slices([conc])
        conc_disc = disc.process_symbol(conc)
        y0 = np.ones(1500)
        jac = conc_disc.jac(y).evaluate(y=y0).toarray()
        np.testing.assert_array_equal(jac, np.eye(1500))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
