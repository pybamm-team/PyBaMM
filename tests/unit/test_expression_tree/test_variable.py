#
# Tests for the Variable class
#
import pybamm
import numpy as np

import unittest


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
