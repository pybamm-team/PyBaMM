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
        self.assertEqual(bin._left, a)
        self.assertEqual(bin._right, b)
        self.assertEqual(a.parent, bin)
        self.assertEqual(b.parent, bin)
        self.assertEqual(bin.children, (a, b))

    def test_addition(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        sum = pybamm.Addition(a, b)
        self.assertEqual(sum._left, a)
        self.assertEqual(sum._right, b)
        self.assertEqual(a.parent, sum)
        self.assertEqual(b.parent, sum)
        self.assertEqual(sum.children, (a, b))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
