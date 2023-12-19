#
# Tests for the Vector class
#
from tests import TestCase
import pybamm
import numpy as np

import unittest


class TestVector(TestCase):
    def setUp(self):
        self.x = np.array([[1], [2], [3]])
        self.vect = pybamm.Vector(self.x)

    def test_array_wrapper(self):
        self.assertEqual(self.vect.ndim, 2)
        self.assertEqual(self.vect.shape, (3, 1))
        self.assertEqual(self.vect.size, 3)

    def test_column_reshape(self):
        vect1d = pybamm.Vector(np.array([1, 2, 3]))
        np.testing.assert_array_equal(self.vect.entries, vect1d.entries)

    def test_list_entries(self):
        vect = pybamm.Vector([1, 2, 3])
        np.testing.assert_array_equal(vect.entries, np.array([[1], [2], [3]]))
        vect = pybamm.Vector([[1], [2], [3]])
        np.testing.assert_array_equal(vect.entries, np.array([[1], [2], [3]]))

    def test_vector_evaluate(self):
        np.testing.assert_array_equal(self.vect.evaluate(), self.x)

    def test_vector_operations(self):
        np.testing.assert_array_equal((self.vect + self.vect).evaluate(), 2 * self.x)
        np.testing.assert_array_equal((self.vect - self.vect).evaluate(), 0 * self.x)
        np.testing.assert_array_equal(
            (self.vect * self.vect).evaluate(), np.array([[1], [4], [9]])
        )

    def test_wrong_size_entries(self):
        with self.assertRaisesRegex(
            ValueError, "Entries must have 1 dimension or be column vector"
        ):
            pybamm.Vector(np.ones((4, 5)))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
