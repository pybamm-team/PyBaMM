#
# Tests for the Parameter class
#

import pytest
import numbers

import pybamm
import sympy


class TestParameter:
    def test_parameter_init(self):
        a = pybamm.Parameter("a")
        assert a.name == "a"
        assert a.domain == []

    def test_evaluate_for_shape(self):
        a = pybamm.Parameter("a")
        assert isinstance(a.evaluate_for_shape(), numbers.Number)

    def test_to_equation(self):
        func = pybamm.Parameter("test_string")
        func1 = pybamm.Parameter("test_name")

        # Test print_name
        func.print_name = "test"
        assert func.to_equation() == sympy.Symbol("test")

        # Test name
        assert func1.to_equation() == sympy.Symbol("test_name")

    def test_to_json_error(self):
        func = pybamm.Parameter("test_string")

        with pytest.raises(NotImplementedError):
            func.to_json()

        with pytest.raises(NotImplementedError):
            pybamm.Parameter._from_json({})


class TestFunctionParameter:
    def test_function_parameter_init(self):
        var = pybamm.Variable("var")
        func = pybamm.FunctionParameter("func", {"var": var})
        assert func.name == "func"
        assert func.children[0] == var
        assert func.domain == []
        assert func.diff_variable is None

    def test_function_parameter_diff(self):
        var = pybamm.Variable("var")
        func = pybamm.FunctionParameter("a", {"var": var}).diff(var)
        assert func.diff_variable == var

    def test_evaluate_for_shape(self):
        a = pybamm.Parameter("a")
        func = pybamm.FunctionParameter("func", {"2a": 2 * a})
        assert isinstance(func.evaluate_for_shape(), numbers.Number)

    def test_copy(self):
        a = pybamm.Parameter("a")
        func = pybamm.FunctionParameter("func", {"2a": 2 * a})

        new_func = func.create_copy()
        assert func.input_names == new_func.input_names

    def test_print_input_names(self):
        var = pybamm.Variable("var")
        func = pybamm.FunctionParameter("a", {"var": var})
        func.print_input_names()

    def test_get_children_domains(self):
        var = pybamm.Variable("var", domain=["negative electrode"])
        var_2 = pybamm.Variable("var", domain=["positive electrode"])
        with pytest.raises(pybamm.DomainError):
            pybamm.FunctionParameter("a", {"var": var, "var 2": var_2})

    def test_set_input_names(self):
        var = pybamm.Variable("var")
        func = pybamm.FunctionParameter("a", {"var": var})

        new_input_names = ["first", "second"]
        func.input_names = new_input_names

        assert func.input_names == new_input_names

        with pytest.raises(TypeError):
            new_input_names = {"wrong": "input type"}
            func.input_names = new_input_names

        with pytest.raises(TypeError):
            new_input_names = [var]
            func.input_names = new_input_names

    def test_print_name(self):
        def myfun(x):
            return pybamm.FunctionParameter("my function", {"x": x})

        def _myfun(x):
            return pybamm.FunctionParameter("my function", {"x": x})

        x = pybamm.Scalar(1)
        assert myfun(x).print_name == "myfun"
        assert _myfun(x).print_name is None

    def test_function_parameter_to_equation(self):
        func = pybamm.FunctionParameter("test", {"x": pybamm.Scalar(1)})
        func1 = pybamm.FunctionParameter("func", {"var": pybamm.Variable("var")})

        # Test print_name
        func.print_name = "test"
        assert func.to_equation() == sympy.Symbol("test")

        # Test name
        func1.print_name = None
        assert func1.to_equation() == sympy.Symbol("func")

    def test_to_json_error(self):
        func = pybamm.FunctionParameter("test", {"x": pybamm.Scalar(1)})

        with pytest.raises(NotImplementedError):
            func.to_json()

        with pytest.raises(NotImplementedError):
            pybamm.FunctionParameter._from_json({})
