#
# SymPy overrides
#
import re

from pybamm.util import have_optional_dependency


LatexPrinter = have_optional_dependency("sympy.printing.latex","LatexPrinter")
class CustomPrint(LatexPrinter):
    """Override SymPy methods to match PyBaMM's requirements"""

    def _print_Derivative(self, expr):
        """Override :meth:`sympy.printing.latex.LatexPrinter._print_Derivative`"""
        eqn = super()._print_Derivative(expr)

        if getattr(expr, "force_partial", False) and "partial" not in eqn:
            var1, var2 = re.findall(r"^\\frac{(\w+)}{(\w+) .+", eqn)[0]
            eqn = eqn.replace(var1, "\partial").replace(var2, "\partial")

        return eqn


def custom_print_func(expr, **settings):
    have_optional_dependency("sympy.printing.latex","LatexPrinter")
    return CustomPrint().doprint(expr)
