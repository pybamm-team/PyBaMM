#
# Tests for the Parameter class
#
from tests import TestCase
import unittest


import pybamm
from pybamm.util import have_optional_dependency


class TestIndependentVariable(TestCase):
    def test_variable_init(self):
        a = pybamm.IndependentVariable("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.domain, [])
        a = pybamm.IndependentVariable("a", domain=["test"])
        self.assertEqual(a.domain[0], "test")
        a = pybamm.IndependentVariable("a", domain="test")
        self.assertEqual(a.domain[0], "test")
        with self.assertRaises(TypeError):
            pybamm.IndependentVariable("a", domain=1)

    def test_time(self):
        t = pybamm.Time()
        self.assertEqual(t.name, "time")
        self.assertEqual(t.evaluate(4), 4)
        with self.assertRaises(ValueError):
            t.evaluate(None)

        t = pybamm.t
        self.assertEqual(t.name, "time")
        self.assertEqual(t.evaluate(4), 4)
        with self.assertRaises(ValueError):
            t.evaluate(None)

        self.assertEqual(t.evaluate_for_shape(), 0)

    def test_spatial_variable(self):
        x = pybamm.SpatialVariable("x", "negative electrode")
        self.assertEqual(x.name, "x")
        self.assertFalse(x.evaluates_on_edges("primary"))
        y = pybamm.SpatialVariable("y", "separator")
        self.assertEqual(y.name, "y")
        z = pybamm.SpatialVariable("z", "positive electrode")
        self.assertEqual(z.name, "z")
        r = pybamm.SpatialVariable("r", "negative particle")
        self.assertEqual(r.name, "r")
        with self.assertRaises(NotImplementedError):
            x.evaluate()

        with self.assertRaisesRegex(ValueError, "domain must be"):
            pybamm.SpatialVariable("x", [])
        with self.assertRaises(pybamm.DomainError):
            pybamm.SpatialVariable("r_n", ["positive particle"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.SpatialVariable("r_p", ["negative particle"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.SpatialVariable("x", ["negative particle"])

    def test_spatial_variable_edge(self):
        x = pybamm.SpatialVariableEdge("x", "negative electrode")
        self.assertEqual(x.name, "x")
        self.assertTrue(x.evaluates_on_edges("primary"))

    def test_to_equation(self):
        sympy = have_optional_dependency("sympy")
        # Test print_name
        func = pybamm.IndependentVariable("a")
        func.print_name = "test"
        self.assertEqual(func.to_equation(), sympy.Symbol("test"))

        self.assertEqual(
            pybamm.IndependentVariable("a").to_equation(), sympy.Symbol("a")
        )

        # Test time
        self.assertEqual(pybamm.t.to_equation(), sympy.Symbol("t"))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
