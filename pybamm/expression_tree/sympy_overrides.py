#
# SymPy overrides
#
from sympy.core.function import _coeff_isneg
from sympy.printing.conventions import requires_partial
from sympy.printing.latex import LatexPrinter
from sympy.printing.precedence import PRECEDENCE


class CustomPrint(LatexPrinter):
    """Override SymPy methods to match PyBaMM's requirements"""

    def _print_Derivative(self, expr):
        """Override :meth:`sympy.printing.latex.LatexPrinter._print_Derivative`"""
        if requires_partial(expr.expr) or getattr(expr, "force_partial", False):
            diff_symbol = r'\partial'
        else:
            diff_symbol = r'd'

        tex = ""
        dim = 0
        for x, num in reversed(expr.variable_count):
            dim += num
            if num == 1:
                tex += r"%s %s" % (diff_symbol, self._print(x))
            else:
                tex += r"%s %s^{%s}" % (diff_symbol,
                                        self.parenthesize_super(self._print(x)),
                                        self._print(num))

        if dim == 1:
            tex = r"\frac{%s}{%s}" % (diff_symbol, tex)
        else:
            tex = r"\frac{%s^{%s}}{%s}" % (diff_symbol, self._print(dim), tex)

        if any(_coeff_isneg(i) for i in expr.args):
            return r"%s %s" % (tex, self.parenthesize(expr.expr,
                                                      PRECEDENCE["Mul"],
                                                      is_neg=True,
                                                      strict=True))

        return r"%s %s" % (tex, self.parenthesize(expr.expr,
                                                  PRECEDENCE["Mul"],
                                                  is_neg=False,
                                                  strict=True))


def custom_print_func(expr, **settings):
    return CustomPrint().doprint(expr)
