#
# Tests for the Vector class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import numpy as np

import unittest


class TestVector(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3])
        self.vect = pybamm.Vector(self.x)

    def test_array_wrapper(self):
        self.assertEqual(self.vect.ndim, 1)
        self.assertEqual(self.vect.shape, (3,))
        self.assertEqual(self.vect.size, 3)

    def test_vector_evaluate(self):
        np.testing.assert_array_equal(self.vect.evaluate(), self.x)

    def test_vector_operations(self):
        np.testing.assert_array_equal((self.vect + self.vect).evaluate(), 2 * self.x)
        np.testing.assert_array_equal((self.vect - self.vect).evaluate(), 0 * self.x)
        np.testing.assert_array_equal(
            (self.vect * self.vect).evaluate(), np.array([1, 4, 9])
        )

    def test_vector_modification(self):
        exp = self.vect * self.vect + self.vect
        self.x[0] = -1
        self.assertTrue(exp.children[1]._entries[0], -1)
        self.assertTrue(exp.children[0].children[0]._entries[0], -1)
        self.assertTrue(exp.children[0].children[1]._entries[0], -1)


class TestStateVector(unittest.TestCase):
    def test_evaluate(self):
        sv = pybamm.StateVector(slice(0, 10))
        y = np.linspace(0, 2, 19)
        np.testing.assert_array_equal(sv.evaluate(y=y), np.linspace(0, 1, 10))

        # Try evaluating with a y that is too short
        y2 = np.ones(5)
        with self.assertRaisesRegex(
            ValueError, "y is too short, so value with slice is smaller than expected"
        ):
            sv.evaluate(y=y2)

    def test_size_and_shape(self):
        sv = pybamm.StateVector(slice(0, 14))
        self.assertEqual(sv.shape, (14,))
        self.assertEqual(sv.size, 14)

        sv = pybamm.StateVector(slice(52, 64))
        self.assertEqual(sv.shape, (12,))
        self.assertEqual(sv.size, 12)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
