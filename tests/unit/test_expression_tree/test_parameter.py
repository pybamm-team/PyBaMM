#
# Tests for the Parameter class
#
import numbers
import pybamm
import unittest


class TestParameter(unittest.TestCase):
    def test_parameter_init(self):
        a = pybamm.Parameter("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.domain, [])
        c = pybamm.Parameter("c", domain=["test"])
        self.assertEqual(c.domain[0], "test")

    def test_evaluate_for_shape(self):
        a = pybamm.Parameter("a")
        self.assertIsInstance(a.evaluate_for_shape(), numbers.Number)


class TestFunctionParameter(unittest.TestCase):
    def test_function_parameter_init(self):
        var = pybamm.Variable("var")
        func = pybamm.FunctionParameter("func", {"var": var})
        self.assertEqual(func.name, "func")
        self.assertEqual(func.children[0].id, var.id)
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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
