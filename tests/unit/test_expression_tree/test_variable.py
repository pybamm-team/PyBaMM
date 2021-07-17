#
# Tests for the Variable class
#
import unittest

import numpy as np
import sympy

import pybamm


class TestVariable(unittest.TestCase):
    def test_variable_init(self):
        a = pybamm.Variable("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.domain, [])
        a = pybamm.Variable("a", domain=["test"])
        self.assertEqual(a.domain[0], "test")
        self.assertRaises(TypeError, pybamm.Variable("a", domain="test"))

    def test_variable_diff(self):
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        self.assertIsInstance(a.diff(a), pybamm.Scalar)
        self.assertEqual(a.diff(a).evaluate(), 1)
        self.assertIsInstance(a.diff(b), pybamm.Scalar)
        self.assertEqual(a.diff(b).evaluate(), 0)

    def test_variable_id(self):
        a1 = pybamm.Variable("a", domain=["negative electrode"])
        a2 = pybamm.Variable("a", domain=["negative electrode"])
        self.assertEqual(a1.id, a2.id)
        a3 = pybamm.Variable("b", domain=["negative electrode"])
        a4 = pybamm.Variable("a", domain=["positive electrode"])
        self.assertNotEqual(a1.id, a3.id)
        self.assertNotEqual(a1.id, a4.id)

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
        self.assertEqual(func.to_equation(), sympy.symbols("test"))

        # Test name
        self.assertEqual(pybamm.Variable("name").to_equation(), "name")


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
        self.assertEqual(a1.id, a2.id)
        a3 = pybamm.VariableDot("b", domain=["negative electrode"])
        a4 = pybamm.VariableDot("a", domain=["positive electrode"])
        self.assertNotEqual(a1.id, a3.id)
        self.assertNotEqual(a1.id, a4.id)

    def test_variable_diff(self):
        a = pybamm.VariableDot("a")
        b = pybamm.Variable("b")
        self.assertIsInstance(a.diff(a), pybamm.Scalar)
        self.assertEqual(a.diff(a).evaluate(), 1)
        self.assertIsInstance(a.diff(b), pybamm.Scalar)
        self.assertEqual(a.diff(b).evaluate(), 0)


class TestExternalVariable(unittest.TestCase):
    def test_external_variable_scalar(self):
        a = pybamm.ExternalVariable("a", 1)
        self.assertEqual(a.size, 1)

        self.assertEqual(a.evaluate(inputs={"a": 3}), 3)

        with self.assertRaisesRegex(KeyError, "External variable"):
            a.evaluate()
        with self.assertRaisesRegex(TypeError, "inputs should be a dictionary"):
            a.evaluate(inputs="not a dictionary")

    def test_external_variable_vector(self):
        a = pybamm.ExternalVariable("a", 10)
        self.assertEqual(a.size, 10)

        a_test = 2 * np.ones((10, 1))
        np.testing.assert_array_equal(a.evaluate(inputs={"a": a_test}), a_test)
        np.testing.assert_array_equal(
            a.evaluate(inputs={"a": a_test.flatten()}), a_test
        )

        np.testing.assert_array_equal(a.evaluate(inputs={"a": 2}), a_test)

        with self.assertRaisesRegex(ValueError, "External variable"):
            a.evaluate(inputs={"a": np.ones((5, 1))})

    def test_external_variable_diff(self):
        a = pybamm.ExternalVariable("a", 10)
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
