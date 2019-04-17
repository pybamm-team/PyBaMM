#
# Tests for the symbolic differentiation methods
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import autograd.numpy as np
import unittest


class TestSymbolicDifferentiation(unittest.TestCase):
    def test_advanced(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        y = np.array([5, 3])

        #
        func = (a * 2 + 5 * (-b)) / (a * b)
        self.assertEqual(func.diff(a).evaluate(y=y), 1 / 5)
        self.assertEqual(func.diff(b).evaluate(y=y), -2 / 9)
        #
        func = a * b ** a
        self.assertAlmostEqual(
            func.diff(a).evaluate(y=y)[0], 3 ** 5 * (5 * np.log(3) + 1)
        )
        self.assertEqual(func.diff(b).evaluate(y=y), 5 ** 2 * 3 ** 4)

    def test_advanced_functions(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        y = np.array([5, 3])

        #
        func = a * pybamm.Function(np.exp, b)
        self.assertAlmostEqual(func.diff(a).evaluate(y=y)[0], np.exp(3))
        func = pybamm.Function(np.exp, a + 2 * b + a * b) + a * pybamm.Function(
            np.exp, b
        )
        self.assertEqual(
            func.diff(a).evaluate(y=y), (4 * np.exp(3 * 5 + 5 + 2 * 3) + np.exp(3))
        )
        self.assertEqual(
            func.diff(b).evaluate(y=y), np.exp(3) * (7 * np.exp(3 * 5 + 5 + 3) + 5)
        )
        #
        func = pybamm.Function(
            np.sin, pybamm.Function(np.cos, a * 4) / 2
        ) * pybamm.Function(np.cos, 4 * pybamm.Function(np.exp, b / 3))
        self.assertEqual(
            func.diff(a).evaluate(y=y),
            -2 * np.sin(20) * np.cos(np.cos(20) / 2) * np.cos(4 * np.exp(1)),
        )
        self.assertEqual(
            func.diff(b).evaluate(y=y),
            -4 / 3 * np.exp(1) * np.sin(4 * np.exp(1)) * np.sin(np.cos(20) / 2),
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
