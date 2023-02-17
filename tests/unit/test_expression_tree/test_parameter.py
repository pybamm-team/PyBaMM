#
# Tests for the Parameter class
#
import numbers
import unittest

import sympy

import pybamm


class TestParameter(unittest.TestCase):
    def test_parameter_init(self):
        a = pybamm.Parameter("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.domain, [])

    def test_evaluate_for_shape(self):
        a = pybamm.Parameter("a")
        self.assertIsInstance(a.evaluate_for_shape(), numbers.Number)

    def test_to_equation(self):
        func = pybamm.Parameter("test_string")
        func1 = pybamm.Parameter("test_name")

        # Test print_name
        func.print_name = "test"
        self.assertEqual(func.to_equation(), sympy.Symbol("test"))

        # Test name
        self.assertEqual(func1.to_equation(), sympy.Symbol("test_name"))


class TestFunctionParameter(unittest.TestCase):
    def test_function_parameter_init(self):
        var = pybamm.Variable("var")
        func = pybamm.FunctionParameter("func", {"var": var})
        self.assertEqual(func.name, "func")
        self.assertEqual(func.children[0], var)
        self.assertEqual(func.domain, [])
        self.assertEqual(func.diff_variable, None)

    def test_function_parameter_diff(self):
        var = pybamm.Variable("var")
        func = pybamm.FunctionParameter("a", {"var": var}).diff(var)
        self.assertEqual(func.diff_variable, var)

    def test_evaluate_for_shape(self):
        a = pybamm.Parameter("a")
        func = pybamm.FunctionParameter("func", {"2a": 2 * a})
        self.assertIsInstance(func.evaluate_for_shape(), numbers.Number)

    def test_copy(self):
        a = pybamm.Parameter("a")
        func = pybamm.FunctionParameter("func", {"2a": 2 * a})

        new_func = func.new_copy()
        self.assertEqual(func.input_names, new_func.input_names)

    def test_print_input_names(self):
        var = pybamm.Variable("var")
        func = pybamm.FunctionParameter("a", {"var": var})
        func.print_input_names()

    def test_get_children_domains(self):
        var = pybamm.Variable("var", domain=["negative electrode"])
        var_2 = pybamm.Variable("var", domain=["positive electrode"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.FunctionParameter("a", {"var": var, "var 2": var_2})

    def test_set_input_names(self):
        var = pybamm.Variable("var")
        func = pybamm.FunctionParameter("a", {"var": var})

        new_input_names = ["first", "second"]
        func.input_names = new_input_names

        self.assertEqual(func.input_names, new_input_names)

        with self.assertRaises(TypeError):
            new_input_names = {"wrong": "input type"}
            func.input_names = new_input_names

        with self.assertRaises(TypeError):
            new_input_names = [var]
            func.input_names = new_input_names

    def test_print_name(self):
        def myfun(x):
            return pybamm.FunctionParameter("my function", {"x": x})

        def _myfun(x):
            return pybamm.FunctionParameter("my function", {"x": x})

        x = pybamm.Scalar(1)
        self.assertEqual(myfun(x).print_name, "myfun")
        self.assertEqual(_myfun(x).print_name, None)

    def test_function_parameter_to_equation(self):
        func = pybamm.FunctionParameter("test", {"x": pybamm.Scalar(1)})
        func1 = pybamm.FunctionParameter("func", {"var": pybamm.Variable("var")})

        # Test print_name
        func.print_name = "test"
        self.assertEqual(func.to_equation(), sympy.Symbol("test"))

        # Test name
        func1.print_name = None
        self.assertEqual(func1.to_equation(), sympy.Symbol("func"))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
