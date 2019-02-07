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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
