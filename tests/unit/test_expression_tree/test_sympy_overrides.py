"""
Tests for the sympy_overrides.py
"""
import unittest

import sympy

import pybamm
from pybamm.expression_tree.sympy_overrides import custom_print_func


class TestCustomPrint(unittest.TestCase):
    def test_print_Derivative(self):
        # Test force_partial
        der1 = sympy.Derivative("y", "x")
        der1.force_partial = True
        self.assertEqual(custom_print_func(der1), "\\frac{\\partial}{\\partial x} y")

        # Test derivative
        der2 = sympy.Derivative("x")
        self.assertEqual(custom_print_func(der2), "\\frac{d}{d x} x")

        # Test partial derivative
        x, y = sympy.symbols("x y")
        expr = x ** 2 + 2 * y + y ** 3
        der3 = sympy.Derivative(expr, x)
        self.assertEqual(
            custom_print_func(der3),
            "\\frac{\\partial}{\\partial x} \\left(x^{2} + y^{3} + 2 y\\right)",
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
