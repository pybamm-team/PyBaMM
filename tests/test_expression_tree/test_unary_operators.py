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
        self.assertEqual(un._child, a)
        self.assertEqual(a.parent, un)
        self.assertEqual(un.children, (a,))

    def test_gradient(self):
        a = pybamm.Symbol("a")
        grad = pybamm.Gradient(a)
        self.assertEqual(grad._child, a)
        self.assertEqual(a.parent, grad)
        self.assertEqual(grad.children, (a,))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
