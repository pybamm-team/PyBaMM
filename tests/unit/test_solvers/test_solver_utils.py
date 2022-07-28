#
# Tests for the solver utility functions and classes
#
import json
import pybamm
import unittest
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tests import get_discretisation_for_testing


class TestSolverUtils(unittest.TestCase):
    def test_compare_numpy_vertcat(self):
        a0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        a1 = np.array([[1, 2, 3]])
        b0 = np.array([[13, 14, 15], [16, 17, 18]])

        for a, b in zip([a0, b0], [a1, b0]):
            pybamm_vertcat = pybamm.NoMemAllocVertcat(a, b)
            np_vertcat = np.concatenate((a, b), axis=0)
            self.assertEqual(pybamm_vertcat.shape, np_vertcat.shape)
            self.assertEqual(pybamm_vertcat.size, np_vertcat.size)
            for i in range(pybamm_vertcat.shape[0]):
                for j in range(pybamm_vertcat.shape[1]):
                    self.assertEqual(pybamm_vertcat[i, j], np_vertcat[i, j])
                    self.assertEqual(pybamm_vertcat[:, j][i], np_vertcat[:, j][i])
            for i in range(pybamm_vertcat.shape[0]):
                np.testing.assert_array_equal(pybamm_vertcat[i, :], np_vertcat[i, :])

    def test_errors(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.ones((4, 5, 6))
        with self.assertRaisesRegex(ValueError, "Only 1D or 2D arrays are supported"):
            pybamm.NoMemAllocVertcat(a, b)

        b = np.array([[10, 11], [13, 14]])
        with self.assertRaisesRegex(
            ValueError, "All arrays must have the same number of columns"
        ):
            pybamm.NoMemAllocVertcat(a, b)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
