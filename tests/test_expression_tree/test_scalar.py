#
# Tests for the Scalar class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestScalar(unittest.TestCase):
    def test_scalar_eval(self):
        a = pybamm.Scalar(5)
        self.assertEqual(a._value, 5)
        self.assertEqual(a.evaluate(None), 5)

    def test_scalar_operations(self):
        a = pybamm.Scalar(5)
        b = pybamm.Scalar(6)
        self.assertEqual((a + b).evaluate(None), 11)
        self.assertEqual((a - b).evaluate(None), -1)
        self.assertEqual((a * b).evaluate(None), 30)
        self.assertEqual((a / b).evaluate(None), 5 / 6)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
