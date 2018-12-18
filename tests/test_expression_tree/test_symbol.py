#
# Test for the Symbol class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
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

        self.assertTrue(isinstance(a + b, pybamm.Symbol))
        self.assertTrue(isinstance(a - b, pybamm.Symbol))
        self.assertTrue(isinstance(a * b, pybamm.Symbol))
        self.assertTrue(isinstance(a / b, pybamm.Symbol))
        with self.assertRaises(NotImplementedError):
            a + 2
        with self.assertRaises(NotImplementedError):
            a - 2
        with self.assertRaises(NotImplementedError):
            a * 2
        with self.assertRaises(NotImplementedError):
            a / 2

    def test_multiple_symbols(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c")
        exp = a * c * (a * b * c + a - c * a)
        expected_preorder = ['*', '*', 'a', 'c', '-', '+', '*', '*',
                             'a', 'b', 'c', 'a', '*', 'c', 'a']
        for node, expect in zip(exp.pre_order(), expected_preorder):
            self.assertEqual(node.name, expect)

    def test_symbol_evaluation(self):
        a = pybamm.Symbol("a")
        with self.assertRaises(NotImplementedError):
            a.evaluate()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
