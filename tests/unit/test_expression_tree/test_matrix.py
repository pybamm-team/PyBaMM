#
# Tests for the Matrix class
#
from tests import TestCase
import pybamm
import numpy as np
from scipy.sparse import csr_matrix

import unittest
import unittest.mock as mock


class TestMatrix(TestCase):
    def setUp(self):
        self.A = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
        self.x = np.array([1, 2, 3])
        self.mat = pybamm.Matrix(self.A)
        self.vect = pybamm.Vector(self.x)

    def test_array_wrapper(self):
        self.assertEqual(self.mat.ndim, 2)
        self.assertEqual(self.mat.shape, (3, 3))
        self.assertEqual(self.mat.size, 9)

    def test_list_entry(self):
        mat = pybamm.Matrix([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(
            mat.entries, np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
        )

    def test_matrix_evaluate(self):
        np.testing.assert_array_equal(self.mat.evaluate(), self.A)

    def test_matrix_operations(self):
        np.testing.assert_array_equal((self.mat + self.mat).evaluate(), 2 * self.A)
        np.testing.assert_array_equal(
            (self.mat - self.mat).evaluate().toarray(), 0 * self.A
        )
        np.testing.assert_array_equal(
            (self.mat @ self.vect).evaluate(), np.array([[5], [2], [3]])
        )

    def test_to_from_json(self):
        arr = pybamm.Matrix(csr_matrix([[0, 1, 0, 0], [0, 0, 0, 1]]))
        json_dict = {
            "name": "Sparse Matrix (2, 4)",
            "id": mock.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "entries": {
                "column_pointers": [0, 1, 2],
                "data": [1.0, 1.0],
                "row_indices": [1, 3],
                "shape": (2, 4),
            },
        }

        self.assertEqual(arr.to_json(), json_dict)

        self.assertEqual(pybamm.Matrix._from_json(json_dict), arr)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
