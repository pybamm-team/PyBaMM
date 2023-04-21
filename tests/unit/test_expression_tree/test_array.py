#
# Tests for the Array class
#
from tests import TestCase
import unittest

import numpy as np
import sympy

import pybamm


class TestArray(TestCase):
    def test_name(self):
        arr = pybamm.Array(np.array([1, 2, 3]))
        self.assertEqual(arr.name, "Array of shape (3, 1)")

    def test_list_entries(self):
        vect = pybamm.Array([1, 2, 3])
        np.testing.assert_array_equal(vect.entries, np.array([[1], [2], [3]]))
        vect = pybamm.Array([[1], [2], [3]])
        np.testing.assert_array_equal(vect.entries, np.array([[1], [2], [3]]))

    def test_linspace(self):
        x = np.linspace(0, 1, 100)[:, np.newaxis]
        y = pybamm.linspace(0, 1, 100)
        np.testing.assert_array_equal(x, y.entries)

    def test_meshgrid(self):
        a = np.linspace(0, 5)
        b = np.linspace(0, 3)
        A, B = np.meshgrid(a, b)
        c = pybamm.linspace(0, 5)
        d = pybamm.linspace(0, 3)
        C, D = pybamm.meshgrid(c, d)
        np.testing.assert_array_equal(A, C.entries)
        np.testing.assert_array_equal(B, D.entries)

    def test_to_equation(self):
        self.assertEqual(
            pybamm.Array([1, 2]).to_equation(), sympy.Array([[1.0], [2.0]])
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
