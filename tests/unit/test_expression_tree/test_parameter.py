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
        func = pybamm.FunctionParameter("func", var)
        self.assertEqual(func.name, "func")
        self.assertEqual(func.children[0].id, var.id)
        self.assertEqual(func.domain, [])
        self.assertEqual(func.diff_variable, None)

    def test_function_parameter_diff(self):
        var = pybamm.Variable("var")
        func = pybamm.FunctionParameter("a", var).diff(var)
        self.assertEqual(func.diff_variable, var)

    def test_evaluate_for_shape(self):
        a = pybamm.Parameter("a")
        func = pybamm.FunctionParameter("func", 2 * a)
        self.assertIsInstance(func.evaluate_for_shape(), numbers.Number)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
