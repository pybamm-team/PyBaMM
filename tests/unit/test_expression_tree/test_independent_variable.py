#
# Tests for the Parameter class
#
import pytest

import pybamm
import sympy


class TestIndependentVariable:
    def test_variable_init(self):
        a = pybamm.IndependentVariable("a")
        assert a.name == "a"
        assert a.domain == []
        a = pybamm.IndependentVariable("a", domain=["test"])
        assert a.domain[0] == "test"
        a = pybamm.IndependentVariable("a", domain="test")
        assert a.domain[0] == "test"
        with pytest.raises(TypeError):
            pybamm.IndependentVariable("a", domain=1)

    def test_time(self):
        t = pybamm.Time()
        assert t.name == "time"
        assert t.evaluate(4) == 4
        with pytest.raises(ValueError):
            t.evaluate(None)

        t = pybamm.t
        assert t.name == "time"
        assert t.evaluate(4) == 4
        with pytest.raises(ValueError):
            t.evaluate(None)

        assert t.evaluate_for_shape() == 0

    def test_spatial_variable(self):
        x = pybamm.SpatialVariable("negative electrode")
        assert not x.evaluates_on_edges("primary")
        with pytest.raises(NotImplementedError):
            x.evaluate()

    def test_spatial_variable_edge(self):
        x = pybamm.SpatialVariableEdge("negative electrode")
        assert x.evaluates_on_edges("primary")

    def test_to_equation(self):
        # Test print_name
        func = pybamm.IndependentVariable("a")
        func.print_name = "test"
        assert func.to_equation() == sympy.Symbol("test")

        assert pybamm.IndependentVariable("a").to_equation() == sympy.Symbol("a")

        # Test time
        assert pybamm.t.to_equation() == sympy.Symbol("t")
