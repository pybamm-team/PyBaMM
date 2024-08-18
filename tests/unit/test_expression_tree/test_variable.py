#
# Tests for the Variable class
#

import pytest

import numpy as np

import pybamm
import sympy


class TestVariable:
    def test_variable_init(self):
        a = pybamm.Variable("a")
        assert a.name == "a"
        assert a.domain == []
        a = pybamm.Variable("a", domain=["test"])
        assert a.domain[0] == "test"
        pytest.raises(TypeError, match=pybamm.Variable("a", domain="test"))
        assert a.scale == 1
        assert a.reference == 0

        a = pybamm.Variable("a", scale=2, reference=-1)
        assert a.scale == 2
        assert a.reference == -1

    def test_variable_diff(self):
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        assert isinstance(a.diff(a), pybamm.Scalar)
        assert a.diff(a).evaluate() == 1
        assert isinstance(a.diff(b), pybamm.Scalar)
        assert a.diff(b).evaluate() == 0

    def test_variable_eq(self):
        a1 = pybamm.Variable("a", domain=["negative electrode"])
        a2 = pybamm.Variable("a", domain=["negative electrode"])
        assert a1 == a2
        a3 = pybamm.Variable("b", domain=["negative electrode"])
        a4 = pybamm.Variable("a", domain=["positive electrode"])
        assert a1 != a3
        assert a1 != a4

    def test_variable_bounds(self):
        var = pybamm.Variable("var")
        assert var.bounds == (-np.inf, np.inf)

        var = pybamm.Variable("var", bounds=(0, 1))
        assert var.bounds == (0, 1)

        with pytest.raises(ValueError, match="Invalid bounds"):
            pybamm.Variable("var", bounds=(1, 0))
        with pytest.raises(ValueError, match="Invalid bounds"):
            pybamm.Variable("var", bounds=(1, 1))

    def test_to_equation(self):
        # Test print_name
        func = pybamm.Variable("test_string")
        func.print_name = "test"
        assert func.to_equation() == sympy.Symbol("test")

        # Test name
        assert pybamm.Variable("name").to_equation() == sympy.Symbol("name")

    def test_to_json_error(self):
        func = pybamm.Variable("test_string")
        with pytest.raises(NotImplementedError):
            func.to_json()


class TestVariableDot:
    def test_variable_init(self):
        a = pybamm.VariableDot("a'")
        assert a.name == "a'"
        assert a.domain == []
        a = pybamm.VariableDot("a", domain=["test"])
        assert a.domain[0] == "test"
        pytest.raises(TypeError, match=pybamm.Variable("a", domain="test"))

    def test_variable_id(self):
        a1 = pybamm.VariableDot("a", domain=["negative electrode"])
        a2 = pybamm.VariableDot("a", domain=["negative electrode"])
        assert a1 == a2
        a3 = pybamm.VariableDot("b", domain=["negative electrode"])
        a4 = pybamm.VariableDot("a", domain=["positive electrode"])
        assert a1 != a3
        assert a1 != a4

    def test_variable_diff(self):
        a = pybamm.VariableDot("a")
        b = pybamm.Variable("b")
        assert isinstance(a.diff(a), pybamm.Scalar)
        assert a.diff(a).evaluate() == 1
        assert isinstance(a.diff(b), pybamm.Scalar)
        assert a.diff(b).evaluate() == 0
