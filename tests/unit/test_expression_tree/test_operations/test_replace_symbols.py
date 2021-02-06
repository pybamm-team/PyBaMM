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
            replaced_symbol = replacer.process_symbol(symbol_in)
            self.assertEqual(replaced_symbol.id, symbol_out.id)

        var1 = pybamm.Variable("var 1", domain="dom 1")
        var2 = pybamm.Variable("var 2", domain="dom 2")
        var3 = pybamm.Variable("var 3", domain="dom 1")
        conc = pybamm.Concatenation(var1, var2)

        replacer = pybamm.SymbolReplacer({var1: var3})
        replaced_symbol = replacer.process_symbol(conc)
        self.assertEqual(replaced_symbol.id, pybamm.Concatenation(var3, var2).id)

    def test_process_model(self):
        model = pybamm.BaseModel()
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        var1 = pybamm.Variable("var1", domain="test")
        var2 = pybamm.Variable("var2", domain="test")
        model.rhs = {var1: a * pybamm.grad(var1)}
        model.algebraic = {var2: c * var2}
        model.initial_conditions = {var1: b, var2: d}
        model.boundary_conditions = {
            var1: {"left": (c, "Dirichlet"), "right": (d, "Neumann")}
        }
        model.variables = {
            "var1": var1,
            "var2": var2,
            "grad_var1": pybamm.grad(var1),
            "d_var1": d * var1,
        }
        model.timescale = b
        model.length_scales = {"test": c}

        replacer = pybamm.SymbolReplacer(
            {
                pybamm.Parameter("a"): pybamm.Scalar(4),
                pybamm.Parameter("b"): pybamm.Scalar(2),
                pybamm.Parameter("c"): pybamm.Scalar(3),
                pybamm.Parameter("d"): pybamm.Scalar(42),
            }
        )
        replacer.process_model(model)
        # rhs
        var1 = model.variables["var1"]
        self.assertIsInstance(model.rhs[var1], pybamm.Multiplication)
        self.assertIsInstance(model.rhs[var1].children[0], pybamm.Scalar)
        self.assertIsInstance(model.rhs[var1].children[1], pybamm.Gradient)
        self.assertEqual(model.rhs[var1].children[0].value, 4)
        # algebraic
        var2 = model.variables["var2"]
        self.assertIsInstance(model.algebraic[var2], pybamm.Multiplication)
        self.assertIsInstance(model.algebraic[var2].children[0], pybamm.Scalar)
        self.assertIsInstance(model.algebraic[var2].children[1], pybamm.Variable)
        self.assertEqual(model.algebraic[var2].children[0].value, 3)
        # initial conditions
        self.assertIsInstance(model.initial_conditions[var1], pybamm.Scalar)
        self.assertEqual(model.initial_conditions[var1].value, 2)
        # boundary conditions
        bc_key = list(model.boundary_conditions.keys())[0]
        self.assertIsInstance(bc_key, pybamm.Variable)
        bc_value = list(model.boundary_conditions.values())[0]
        self.assertIsInstance(bc_value["left"][0], pybamm.Scalar)
        self.assertEqual(bc_value["left"][0].value, 3)
        self.assertIsInstance(bc_value["right"][0], pybamm.Scalar)
        self.assertEqual(bc_value["right"][0].value, 42)
        # variables
        self.assertEqual(model.variables["var1"].id, var1.id)
        self.assertIsInstance(model.variables["grad_var1"], pybamm.Gradient)
        self.assertTrue(
            isinstance(model.variables["grad_var1"].children[0], pybamm.Variable)
        )
        self.assertEqual(model.variables["d_var1"].id, (pybamm.Scalar(42) * var1).id)
        self.assertIsInstance(model.variables["d_var1"].children[0], pybamm.Scalar)
        self.assertTrue(
            isinstance(model.variables["d_var1"].children[1], pybamm.Variable)
        )
        # timescale and length scales
        self.assertEqual(model.timescale.evaluate(), 2)
        self.assertEqual(model.length_scales["test"].evaluate(), 3)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
