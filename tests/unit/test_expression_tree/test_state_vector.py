#
# Tests for the State Vector class
#
import pybamm
import numpy as np

import unittest


class TestStateVector(unittest.TestCase):
    def test_evaluate(self):
        sv = pybamm.StateVector(slice(0, 10))
        y = np.linspace(0, 2, 19)
        np.testing.assert_array_equal(
            sv.evaluate(y=y), np.linspace(0, 1, 10)[:, np.newaxis]
        )

        # Try evaluating with a y that is too short
        y2 = np.ones(5)
        with self.assertRaisesRegex(
            ValueError, "y is too short, so value with slice is smaller than expected"
        ):
            sv.evaluate(y=y2)

    def test_evaluate_list(self):
        sv = pybamm.StateVector(slice(0, 11), slice(20, 31))
        y = np.linspace(0, 3, 31)
        np.testing.assert_array_almost_equal(
            sv.evaluate(y=y),
            np.concatenate([np.linspace(0, 1, 11), np.linspace(2, 3, 11)])[
                :, np.newaxis
            ],
        )
        sv = pybamm.StateVector(slice(0, 11), slice(11, 20), slice(20, 31))
        y = np.linspace(0, 3, 31)
        np.testing.assert_array_almost_equal(sv.evaluate(y=y), y[:, np.newaxis])

    def test_name(self):
        sv = pybamm.StateVector(slice(0, 10))
        self.assertEqual(sv.name, "y[0:10]")
        sv = pybamm.StateVector(slice(0, 10), slice(20, 30))
        self.assertEqual(sv.name, "y[0:10,20:30]")
        sv = pybamm.StateVector(
            slice(0, 10), slice(20, 30), slice(40, 50), slice(60, 70)
        )
        self.assertEqual(sv.name, "y[0:10,20:30,...,60:70]")

    def test_failure(self):
        with self.assertRaisesRegex(TypeError, "all y_slices must be slice objects"):
            pybamm.StateVector(slice(0, 10), 1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
