#
# Tests for the Variable class
#

import unittest

import numpy as np

import pybamm
import sympy


class TestVariable(unittest.TestCase):
    def test_variable_init(self):
        a = pybamm.Variable("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.domain, [])
        a = pybamm.Variable("a", domain=["test"])
        self.assertEqual(a.domain[0], "test")
        self.assertRaises(TypeError, pybamm.Variable("a", domain="test"))
        self.assertEqual(a.scale, 1)
        self.assertEqual(a.reference, 0)

        a = pybamm.Variable("a", scale=2, reference=-1)
        self.assertEqual(a.scale, 2)
        self.assertEqual(a.reference, -1)

    def test_variable_diff(self):
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        self.assertIsInstance(a.diff(a), pybamm.Scalar)
        self.assertEqual(a.diff(a).evaluate(), 1)
        self.assertIsInstance(a.diff(b), pybamm.Scalar)
        self.assertEqual(a.diff(b).evaluate(), 0)

    def test_variable_eq(self):
        a1 = pybamm.Variable("a", domain=["negative electrode"])
        a2 = pybamm.Variable("a", domain=["negative electrode"])
        self.assertEqual(a1, a2)
        a3 = pybamm.Variable("b", domain=["negative electrode"])
        a4 = pybamm.Variable("a", domain=["positive electrode"])
        self.assertNotEqual(a1, a3)
        self.assertNotEqual(a1, a4)

    def test_variable_bounds(self):
        var = pybamm.Variable("var")
        self.assertEqual(var.bounds, (-np.inf, np.inf))

        var = pybamm.Variable("var", bounds=(0, 1))
        self.assertEqual(var.bounds, (0, 1))

        with self.assertRaisesRegex(ValueError, "Invalid bounds"):
            pybamm.Variable("var", bounds=(1, 0))
        with self.assertRaisesRegex(ValueError, "Invalid bounds"):
            pybamm.Variable("var", bounds=(1, 1))

    def test_to_equation(self):
        # Test print_name
        func = pybamm.Variable("test_string")
        func.print_name = "test"
        self.assertEqual(func.to_equation(), sympy.Symbol("test"))

        # Test name
        self.assertEqual(pybamm.Variable("name").to_equation(), sympy.Symbol("name"))

    def test_to_json_error(self):
        func = pybamm.Variable("test_string")
        with self.assertRaises(NotImplementedError):
            func.to_json()


class TestVariableDot(unittest.TestCase):
    def test_variable_init(self):
        a = pybamm.VariableDot("a'")
        self.assertEqual(a.name, "a'")
        self.assertEqual(a.domain, [])
        a = pybamm.VariableDot("a", domain=["test"])
        self.assertEqual(a.domain[0], "test")
        self.assertRaises(TypeError, pybamm.Variable("a", domain="test"))

    def test_variable_id(self):
        a1 = pybamm.VariableDot("a", domain=["negative electrode"])
        a2 = pybamm.VariableDot("a", domain=["negative electrode"])
        self.assertEqual(a1, a2)
        a3 = pybamm.VariableDot("b", domain=["negative electrode"])
        a4 = pybamm.VariableDot("a", domain=["positive electrode"])
        self.assertNotEqual(a1, a3)
        self.assertNotEqual(a1, a4)

    def test_variable_diff(self):
        a = pybamm.VariableDot("a")
        b = pybamm.Variable("b")
        self.assertIsInstance(a.diff(a), pybamm.Scalar)
        self.assertEqual(a.diff(a).evaluate(), 1)
        self.assertIsInstance(a.diff(b), pybamm.Scalar)
        self.assertEqual(a.diff(b).evaluate(), 0)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
