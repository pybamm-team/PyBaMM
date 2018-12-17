#
# Tests for the Unary Operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestUnaryOperators(unittest.TestCase):
    def test_unary_operator(self):
        a = pybamm.Symbol("a")
        un = pybamm.UnaryOperator("unary test", a)
        self.assertEqual(un.children[0].name, a.name)

    def test_gradient(self):
        a = pybamm.Symbol("a")
        grad = pybamm.Gradient(a)
        self.assertEqual(grad.children[0].name, a.name)

    def test_gradient_printing(self):
        a = pybamm.Symbol("a")
        grad = pybamm.Gradient(a)
        self.assertEqual(grad.name, "grad")
        self.assertEqual(str(grad), "grad(a)")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
