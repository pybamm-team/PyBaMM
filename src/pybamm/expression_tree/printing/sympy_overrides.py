#
# SymPy overrides
#
import re

import sympy
from sympy.printing.latex import LatexPrinter

# LaTeX structural commands that indicate a Symbol name is already
# pre-formatted LaTeX (not a simple variable name from prettify_print_name).
_PREFORMATTED_MARKERS = frozenset(
    [
        r"\frac",
        r"\text",
        r"\quad",
        r"\nabla",
        r"\left",
        r"\right",
        r"\begin",
        r"\end",
        r"\cdot",
        r"\sqrt",
        r"\sum",
        r"\prod",
        r"\int",
        r"\lim",
        r"\cases",
        r"\partial",
    ]
)


class CustomPrint(LatexPrinter):
    """Override SymPy methods to match PyBaMM's requirements"""

    def _print_Derivative(self, expr):
        """Override :meth:`sympy.printing.latex.LatexPrinter._print_Derivative`"""
        eqn = super()._print_Derivative(expr)
        if getattr(expr, "force_partial", False) and "partial" not in eqn:
            var1, var2 = re.findall(r"^\\frac{(\w+)}{(\w+) .+", eqn)[0]
            eqn = eqn.replace(var1, r"\partial").replace(var2, r"\partial")

        return eqn

    def _print_Symbol(self, expr, style="plain"):
        """Return pre-formatted LaTeX Symbol names verbatim.

        PyBaMM's ``latexify`` wraps already-rendered LaTeX strings inside
        ``sympy.Symbol(...)``.  The parent ``_print_Symbol`` passes the name
        through ``split_super_sub`` which re-parses ``_`` / ``^`` characters
        and can mangle the LaTeX.  Detect such cases and return the name as-is.
        """
        name = expr.name
        if any(marker in name for marker in _PREFORMATTED_MARKERS):
            return name
        return super()._print_Symbol(expr, style=style)

    def _print_Pow(self, expr):
        """Avoid double-superscript when the rendered base already contains ``^``.

        This can happen in two ways:

        1. ``prettify_print_name`` encodes modifiers like *typ*, *init*, *surf*
           as superscripts (e.g. ``R_{\\mathrm{n}}^{\\mathrm{typ}}``).
        2. A complex sub-expression (e.g. ``(1 - \\epsilon)^{b}``) is itself
           the base of an outer power.

        In both cases, SymPy's default template ``{base}^{exp}`` produces
        LaTeX that KaTeX rejects as "Double superscript".  We detect the
        conflict by rendering the base first and checking for ``^``.
        """
        base_latex = self._print(expr.base)

        if "^" not in base_latex:
            # No risk of double superscript — use default rendering.
            return super()._print_Pow(expr)

        # Handle common special exponents so we don't lose SymPy niceties.
        exp = expr.exp
        if exp == sympy.S.Half:
            return r"\sqrt{" + base_latex + "}"
        if exp == -sympy.S.Half:
            return r"\frac{1}{\sqrt{" + base_latex + "}}"
        if exp == sympy.S.NegativeOne:
            return r"\frac{1}{" + base_latex + "}"
        if getattr(exp, "is_negative", False) and exp != sympy.S.NegativeOne:
            pos_exp_latex = self._print(-exp)
            return r"\frac{1}{\left(" + base_latex + r"\right)^{" + pos_exp_latex + "}}"

        exp_latex = self._print(exp)
        return r"\left(" + base_latex + r"\right)^{" + exp_latex + "}"


def custom_print_func(expr, **settings):
    return CustomPrint().doprint(expr)
