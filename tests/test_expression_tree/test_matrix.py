#
# Tests for the Matrix class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import numpy as np

import unittest


class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
        self.x = np.array([1, 2, 3])
        self.mat = pybamm.Matrix(self.A)
        self.vect = pybamm.Vector(self.x)

    def test_array_wrapper(self):
        self.assertEqual(self.mat.ndim, 2)
        self.assertEqual(self.mat.shape, (3, 3))
        self.assertEqual(self.mat.size, 9)

    def test_matrix_evaluate(self):
        self.assertTrue((self.mat.evaluate() == self.A).all(), self.A)

    def test_matrix_operations(self):
        self.assertTrue(((self.mat + self.mat).evaluate() == 2 * self.A).all())
        self.assertTrue(((self.mat - self.mat).evaluate() == 0 * self.A).all())
        self.assertTrue(
            ((self.mat * self.vect).evaluate() == np.array([5, 2, 3])).all()
        )

    def test_matrix_modification(self):
        exp = self.mat * self.mat + self.mat
        self.A[0, 0] = -1
        self.assertTrue(exp.children[1]._entries[0, 0], -1)
        self.assertTrue(exp.children[0].children[0]._entries[0, 0], -1)
        self.assertTrue(exp.children[0].children[1]._entries[0, 0], -1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
