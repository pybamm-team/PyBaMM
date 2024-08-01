#
# Tests for the Scalar class
#
import pytest
import pybamm

import numpy as np


class TestDDT:
    def test_time_derivative(self):
        a = pybamm.Scalar(5).diff(pybamm.t)
        assert isinstance(a, pybamm.Scalar)
        assert a.value == 0

        a = pybamm.t.diff(pybamm.t)
        assert isinstance(a, pybamm.Scalar)
        assert a.value == 1

        a = (pybamm.t**2).diff(pybamm.t)
        assert a == (2 * pybamm.t**1 * 1)
        assert a.evaluate(t=1) == 2

        a = (2 + pybamm.t**2).diff(pybamm.t)
        assert a.evaluate(t=1) == 2

    def test_time_derivative_of_variable(self):
        a = (pybamm.Variable("a")).diff(pybamm.t)
        assert isinstance(a, pybamm.VariableDot)
        assert a.name == "a'"

        p = pybamm.Parameter("p")
        a = 1 + p * pybamm.Variable("a")
        diff_a = a.diff(pybamm.t)
        assert isinstance(diff_a, pybamm.Multiplication)
        assert diff_a.children[0].name == "p"
        assert diff_a.children[1].name == "a'"

        with pytest.raises(pybamm.ModelError):
            a = (pybamm.Variable("a")).diff(pybamm.t).diff(pybamm.t)

    def test_time_derivative_of_state_vector(self):
        sv = pybamm.StateVector(slice(0, 10))
        y_dot = np.linspace(0, 2, 19)

        a = sv.diff(pybamm.t)
        assert isinstance(a, pybamm.StateVectorDot)
        assert a.name[-1] == "'"
        np.testing.assert_array_equal(
            a.evaluate(y_dot=y_dot), np.linspace(0, 1, 10)[:, np.newaxis]
        )

        with pytest.raises(pybamm.ModelError):
            a = (sv).diff(pybamm.t).diff(pybamm.t)
