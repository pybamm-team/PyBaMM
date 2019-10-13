#
# Tests for the Function classes
#
import pybamm

import unittest
import numpy as np
import autograd.numpy as auto_np
from scipy.interpolate import interp1d


class TestInterpolant(unittest.TestCase):
    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "data should have exactly two columns"):
            pybamm.Interpolant(np.ones(10), None)
        with self.assertRaisesRegex(ValueError, "interpolator 'bla' not recognised"):
            pybamm.Interpolant(np.ones((10, 2)), None, interpolator="bla")

    def test_interpolation(self):
        x = np.linspace(0, 1)[:, np.newaxis]
        y = pybamm.StateVector(slice(0, 2))
        # linear
        linear = np.hstack([x, 2 * x])
        for interpolator in ["pchip", "cubic spline"]:
            interp = pybamm.Interpolant(linear, y, interpolator=interpolator)
            np.testing.assert_array_almost_equal(
                interp.evaluate(y=np.array([0.397, 1.5]))[:, 0], np.array([0.794, 3])
            )
        # square
        square = np.hstack([x, x ** 2])
        y = pybamm.StateVector(slice(0, 1))
        for interpolator in ["pchip", "cubic spline"]:
            interp = pybamm.Interpolant(square, y, interpolator=interpolator)
            np.testing.assert_array_almost_equal(
                interp.evaluate(y=np.array([0.397]))[:, 0], np.array([0.397 ** 2])
            )

        # with extrapolation set to False
        for interpolator in ["pchip", "cubic spline"]:
            interp = pybamm.Interpolant(
                square, y, interpolator=interpolator, extrapolate=False
            )
            np.testing.assert_array_equal(
                interp.evaluate(y=np.array([2]))[:, 0], np.array([np.nan])
            )

    def test_name(self):
        a = pybamm.Symbol("a")
        x = np.linspace(0, 1)[:, np.newaxis]
        interp = pybamm.Interpolant(np.hstack([x, x]), a, "name")
        self.assertEqual(interp.name, "interpolating function (name)")

    def test_diff(self):
        x = np.linspace(0, 1)[:, np.newaxis]
        y = pybamm.StateVector(slice(0, 2))
        # linear (derivative should be 2)
        linear = np.hstack([x, 2 * x])
        for interpolator in ["pchip", "cubic spline"]:
            interp_diff = pybamm.Interpolant(linear, y, interpolator=interpolator).diff(
                y
            )
            np.testing.assert_array_almost_equal(
                interp_diff.evaluate(y=np.array([0.397, 1.5]))[:, 0], np.array([2, 2])
            )
        # square (derivative should be 2*x)
        square = np.hstack([x, x ** 2])
        for interpolator in ["pchip", "cubic spline"]:
            interp_diff = pybamm.Interpolant(square, y, interpolator=interpolator).diff(
                y
            )
            np.testing.assert_array_almost_equal(
                interp_diff.evaluate(y=np.array([0.397, 0.806]))[:, 0],
                np.array([0.794, 1.612]),
                decimal=3,
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
