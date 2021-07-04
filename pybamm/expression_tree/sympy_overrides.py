#
# SymPy overrides
#
import re

from sympy.printing.latex import LatexPrinter


class CustomPrint(LatexPrinter):
    """Override SymPy methods to match PyBaMM's requirements"""

    def _print_Derivative(self, expr):
        """Override :meth:`sympy.printing.latex.LatexPrinter._print_Derivative`"""
        eqn = super()._print_Derivative(expr)

        if getattr(expr, "force_partial", False) and "partial" not in eqn:
            a1, a2 = re.findall(r"^\\frac{(\w+)}{(\w+) .+", eqn)[0]
            eqn = eqn.replace(a1, "\partial").replace(a2, "\partial")

        return eqn


def custom_print_func(expr, **settings):
    return CustomPrint().doprint(expr)
