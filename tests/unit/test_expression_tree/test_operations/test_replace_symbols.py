#
# Tests for the symbol replacer
#
import pybamm
import unittest


class TestSymbolReplacer(unittest.TestCase):
    def test_symbol_replacements(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        replacer = pybamm.SymbolReplacer({a: b, c: d})

        for symbol_in, symbol_out in [
            (a, b),  # just the symbol
            (a + a, b + b),  # binary operator
            (2 * pybamm.sin(a), 2 * pybamm.sin(b)),  # function
            (3 * b, 3 * b),  # no replacement
            (a + c, b + d),  # two replacements
        ]:
            replaced_symbol = replacer.replace_symbols(symbol_in)
            self.assertEqual(replaced_symbol.id, symbol_out.id)

        var1 = pybamm.Variable("var 1", domain="dom 1")
        var2 = pybamm.Variable("var 2", domain="dom 2")
        var3 = pybamm.Variable("var 3", domain="dom 1")
        conc = pybamm.Concatenation(var1, var2)

        replacer = pybamm.SymbolReplacer({var1: var3})
        replaced_symbol = replacer.replace_symbols(conc)
        self.assertEqual(replaced_symbol.id, pybamm.Concatenation(var3, var2).id)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
