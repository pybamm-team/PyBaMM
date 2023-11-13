#
# SymPy overrides
#
import re

from pybamm.util import have_optional_dependency


def custom_latex_printer(expr, **settings):
    latex = have_optional_dependency("sympy", "latex")
    Derivative = have_optional_dependency("sympy", "Derivative")
    if isinstance(expr, Derivative) and getattr(expr, "force_partial", False):
        latex_str = latex(expr, **settings)
        var1, var2 = re.findall(r"^\\frac{(\w+)}{(\w+) .+", latex_str)[0]
        latex_str = latex_str.replace(var1, "\partial").replace(var2, "\partial")
        return latex_str
    else:
        return latex(expr, **settings)

class CustomPrint:
    """Override SymPy methods to match PyBaMM's requirements"""

    def _print_Derivative(self, expr):
        """Override :meth:`sympy.printing.latex.LatexPrinter._print_Derivative`"""
        return custom_latex_printer(expr)

def custom_print_func(expr, **settings):
    have_optional_dependency("sympy.printing.latex", "LatexPrinter")
    return CustomPrint()._print_Derivative(expr)
