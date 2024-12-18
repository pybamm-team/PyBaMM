#
# Tests for the symbolic differentiation methods
#

import pytest
import numpy as np
import pybamm
from numpy import testing


class TestSymbolicDifferentiation:
    def test_advanced(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        y = np.array([5, 3])

        #
        func = (a * 2 + 5 * (-b)) / (a * b)
        assert func.diff(a).evaluate(y=y) == 1 / 5
        assert func.diff(b).evaluate(y=y) == -2 / 9
        #
        func = a * b**a
        testing.assert_array_almost_equal(
            func.diff(a).evaluate(y=y)[0], 3**5 * (5 * np.log(3) + 1)
        )
        assert func.diff(b).evaluate(y=y) == 5**2 * 3**4

    def test_advanced_functions(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        y = np.array([5, 3])

        #
        func = a * pybamm.exp(b)
        assert func.diff(a).evaluate(y=y)[0] == pytest.approx(np.exp(3))
        func = pybamm.exp(a + 2 * b + a * b) + a * pybamm.exp(b)
        assert func.diff(a).evaluate(y=y) == (4 * np.exp(3 * 5 + 5 + 2 * 3) + np.exp(3))
        assert func.diff(b).evaluate(y=y) == np.exp(3) * (7 * np.exp(3 * 5 + 5 + 3) + 5)
        #
        func = pybamm.sin(pybamm.cos(a * 4) / 2) * pybamm.cos(4 * pybamm.exp(b / 3))
        assert func.diff(a).evaluate(y=y) == -2 * np.sin(20) * np.cos(
            np.cos(20) / 2
        ) * np.cos(4 * np.exp(1))
        assert func.diff(b).evaluate(y=y) == -4 / 3 * np.exp(1) * np.sin(
            4 * np.exp(1)
        ) * np.sin(np.cos(20) / 2)
        #
        func = pybamm.sin(a * b)
        assert func.diff(a).evaluate(y=y) == 3 * np.cos(15)

    def test_diff_zero(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        func = (a * 2 + 5 * (-a)) / (a * a)
        assert func.diff(b) == pybamm.Scalar(0)
        assert func.diff(a) != pybamm.Scalar(0)

    def test_diff_state_vector_dot(self):
        a = pybamm.StateVectorDot(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        assert a.diff(a) == pybamm.Scalar(1)
        assert a.diff(b) == pybamm.Scalar(0)

    def test_diff_heaviside(self):
        a = pybamm.Scalar(1)
        b = pybamm.StateVector(slice(0, 1))

        func = (a < b) * (2 * b)
        assert func.diff(b).evaluate(y=np.array([2])) == 2
        assert func.diff(b).evaluate(y=np.array([-2])) == 0

    def test_diff_modulo(self):
        a = pybamm.Scalar(3)
        b = pybamm.StateVector(slice(0, 1))

        func = (a % b) * (b**2)
        assert func.diff(b).evaluate(y=np.array([2])) == 0
        assert func.diff(b).evaluate(y=np.array([5])) == 30
        assert func.diff(b).evaluate(y=np.array([-2])) == 12

    def test_diff_maximum_minimum(self):
        a = pybamm.Scalar(1)
        b = pybamm.StateVector(slice(0, 1))

        func = pybamm.minimum(a, b**3)
        assert func.diff(b).evaluate(y=np.array([10])) == 0
        assert func.diff(b).evaluate(y=np.array([2])) == 0
        assert func.diff(b).evaluate(y=np.array([-2])) == 3 * (-2) ** 2

        func = pybamm.maximum(a, b**3)
        assert func.diff(b).evaluate(y=np.array([10])) == 3 * 10**2
        assert func.diff(b).evaluate(y=np.array([2])) == 3 * 2**2
        assert func.diff(b).evaluate(y=np.array([-2])) == 0

    def test_exceptions(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        with pytest.raises(NotImplementedError):
            a._diff(b)
