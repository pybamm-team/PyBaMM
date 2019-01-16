#
# Test for the Symbol class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import os
import pybamm

import unittest


class TestSymbol(unittest.TestCase):
    def test_symbol_init(self):
        sym = pybamm.Symbol("a symbol")
        self.assertEqual(sym.name, "a symbol")
        self.assertEqual(str(sym), "a symbol")

    def test_symbol_methods(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")

        # unary
        self.assertTrue(isinstance(-a, pybamm.Negate))
        self.assertTrue(isinstance(abs(a), pybamm.AbsoluteValue))

        # binary - two symbols
        self.assertTrue(isinstance(a + b, pybamm.Addition))
        self.assertTrue(isinstance(a - b, pybamm.Subtraction))
        self.assertTrue(isinstance(a * b, pybamm.Multiplication))
        self.assertTrue(isinstance(a / b, pybamm.Division))
        self.assertTrue(isinstance(a ** b, pybamm.Power))

        # binary - symbol and number
        self.assertTrue(isinstance(a + 2, pybamm.Addition))
        self.assertTrue(isinstance(a - 2, pybamm.Subtraction))
        self.assertTrue(isinstance(a * 2, pybamm.Multiplication))
        self.assertTrue(isinstance(a / 2, pybamm.Division))
        self.assertTrue(isinstance(a ** 2, pybamm.Power))

        # binary - number and symbol
        self.assertTrue(isinstance(3 + b, pybamm.Addition))
        self.assertEqual((3 + b).children[1].id, b.id)
        self.assertTrue(isinstance(3 - b, pybamm.Subtraction))
        self.assertEqual((3 - b).children[1].id, b.id)
        self.assertTrue(isinstance(3 * b, pybamm.Multiplication))
        self.assertEqual((3 * b).children[1].id, b.id)
        self.assertTrue(isinstance(3 / b, pybamm.Division))
        self.assertEqual((3 / b).children[1].id, b.id)
        self.assertTrue(isinstance(3 ** b, pybamm.Power))
        self.assertEqual((3 ** b).children[1].id, b.id)

        # error raising
        with self.assertRaises(NotImplementedError):
            a + "two"
        with self.assertRaises(NotImplementedError):
            a - "two"
        with self.assertRaises(NotImplementedError):
            a * "two"
        with self.assertRaises(NotImplementedError):
            a / "two"
        with self.assertRaises(NotImplementedError):
            a ** "two"
        with self.assertRaises(NotImplementedError):
            "two" + a
        with self.assertRaises(NotImplementedError):
            "two" - a
        with self.assertRaises(NotImplementedError):
            "two" * a
        with self.assertRaises(NotImplementedError):
            "two" / a
        with self.assertRaises(NotImplementedError):
            "two" ** a

    def test_multiple_symbols(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c")
        exp = a * c * (a * b * c + a - c * a)
        expected_preorder = [
            "*",
            "*",
            "a",
            "c",
            "-",
            "+",
            "*",
            "*",
            "a",
            "b",
            "c",
            "a",
            "*",
            "c",
            "a",
        ]
        for node, expect in zip(exp.pre_order(), expected_preorder):
            self.assertEqual(node.name, expect)

    def test_symbol_evaluation(self):
        a = pybamm.Symbol("a")
        with self.assertRaises(NotImplementedError):
            a.evaluate()

    def test_symbol_repr(self):
        """
        test that __repr___ returns the string
        `__class__(id, name, parent expression)`
        """
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        hex_regex = r"\-?0x[0-9,a-f]+"
        self.assertRegex(a.__repr__(), r"Symbol\(" + hex_regex + r", a, None\)")
        self.assertRegex(b.__repr__(), r"Symbol\(" + hex_regex + r", b, None\)")
        self.assertRegex(
            (a + b).__repr__(), r"Addition\(" + hex_regex + r", \+, None\)"
        )
        self.assertRegex(
            (a + b).children[0].__repr__(), r"Symbol\(" + hex_regex + r", a, a \+ b\)"
        )
        self.assertRegex(
            (a + b).children[1].__repr__(), r"Symbol\(" + hex_regex + r", b, a \+ b\)"
        )
        self.assertRegex(
            (a * b).__repr__(), r"Multiplication\(" + hex_regex + r", \*, None\)"
        )
        self.assertRegex(
            (a * b).children[0].__repr__(), r"Symbol\(" + hex_regex + r", a, a \* b\)"
        )
        self.assertRegex(
            (a * b).children[1].__repr__(), r"Symbol\(" + hex_regex + ", b, a \* b\)"
        )
        self.assertRegex(
            pybamm.grad(a).__repr__(), r"Gradient\(" + hex_regex + ", grad, None\)"
        )
        self.assertRegex(
            pybamm.grad(a).children[0].__repr__(),
            r"Symbol\(" + hex_regex + ", a, grad\(a\)\)",
        )

    def test_symbol_visualise(self):
        G = pybamm.Symbol("G")
        model = pybamm.electrolyte.StefanMaxwellDiffusion(G)
        c_e = list(model.rhs.keys())[0]
        rhs = model.rhs[c_e]
        rhs.visualise("StefanMaxwell_test")
        os.remove("view_tree/StefanMaxwell_test.png")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
