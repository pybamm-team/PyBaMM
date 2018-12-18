#
# Tests for the Binary Operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestBinaryOperators(unittest.TestCase):
    def test_binary_operator(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        bin = pybamm.BinaryOperator("binary test", a, b)
        self.assertEqual(bin.children[0].name, a.name)
        self.assertEqual(bin.children[1].name, b.name)

    def test_addition(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        sum = pybamm.Addition(a, b)
        self.assertEqual(sum.children[0].name, a.name)
        self.assertEqual(sum.children[1].name, b.name)

    def test_addition_printing(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        sum = pybamm.Addition(a, b)
        self.assertEqual(sum.name, "+")
        self.assertEqual(str(sum), "a + b")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
