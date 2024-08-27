"""
Tests for the sympy_overrides.py
"""

from pybamm.expression_tree.printing.sympy_overrides import custom_print_func
import sympy


class TestCustomPrint:
    def test_print_derivative(self):
        # Test force_partial
        der1 = sympy.Derivative("y", "x")
        der1.force_partial = True
        assert custom_print_func(der1) == "\\frac{\\partial}{\\partial x} y"

        # Test derivative
        der2 = sympy.Derivative("x")
        assert custom_print_func(der2) == "\\frac{d}{d x} x"
