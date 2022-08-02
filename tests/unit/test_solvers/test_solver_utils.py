#
# Tests for the solver utility functions and classes
#
import pybamm
import unittest
import numpy as np


class TestSolverUtils(unittest.TestCase):
    def test_compare_numpy_vertcat(self):
        x = [np.array([1, 2]), np.array([4, 5]), np.array([7, 8])]
        z = [np.array([13, 14, 15]), np.array([16, 17, 18]), np.array([19, 20, 21])]

        pybamm_vertcat = pybamm.NoMemAllocVertcat(x, z)

        self.assertEqual(pybamm_vertcat.shape, (5, 3))
        np.testing.assert_array_equal(
            pybamm_vertcat[:, 0], np.array([1, 2, 13, 14, 15])[:, np.newaxis]
        )
        np.testing.assert_array_equal(
            pybamm_vertcat[:, -1], np.array([7, 8, 19, 20, 21])[:, np.newaxis]
        )

        pybamm_sub_vertcat = pybamm_vertcat[:, 1:]
        self.assertEqual(pybamm_sub_vertcat.shape, (5, 2))
        np.testing.assert_array_equal(pybamm_vertcat[:, 1], pybamm_sub_vertcat[:, 0])

    def test_errors(self):
        x = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        z = [np.array([13, 14, 15]), np.array([16, 17, 18]), np.array([19, 20, 21])]

        pybamm_vertcat = pybamm.NoMemAllocVertcat(x, z)
        with self.assertRaisesRegex(NotImplementedError, "Only full slices"):
            pybamm_vertcat[0, 0]


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
