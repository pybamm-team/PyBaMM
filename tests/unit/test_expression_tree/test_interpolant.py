#
# Tests for the Function classes
#
import pybamm

import unittest
import numpy as np


class TestInterpolant(unittest.TestCase):
    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "x1"):
            pybamm.Interpolant(np.ones(10), np.ones(11), pybamm.Symbol("a"))
        with self.assertRaisesRegex(ValueError, "x2"):
            pybamm.Interpolant(
                (np.ones(12), np.ones(11)), np.ones((10, 12)), pybamm.Symbol("a")
            )
        with self.assertRaisesRegex(ValueError, "x1"):
            pybamm.Interpolant(
                (np.ones(11), np.ones(10)), np.ones((10, 12)), pybamm.Symbol("a")
            )
        with self.assertRaisesRegex(ValueError, "y should"):
            pybamm.Interpolant(
                (np.ones(10), np.ones(11)), np.ones(10), pybamm.Symbol("a")
            )
        with self.assertRaisesRegex(ValueError, "interpolator 'bla' not recognised"):
            pybamm.Interpolant(
                np.ones(10), np.ones(10), pybamm.Symbol("a"), interpolator="bla"
            )
        with self.assertRaisesRegex(ValueError, "child should have size 1"):
            pybamm.Interpolant(
                np.ones(10), np.ones((10, 11)), pybamm.StateVector(slice(0, 2))
            )
        with self.assertRaisesRegex(ValueError, "should equal"):
            pybamm.Interpolant(
                (np.ones(12), np.ones(10)), np.ones((10, 12)), pybamm.Symbol("a")
            )
        with self.assertRaisesRegex(ValueError, "interpolator should be 'linear'"):
            pybamm.Interpolant(
                (np.ones(10), np.ones(12)),
                np.ones((10, 12)),
                (pybamm.Symbol("a"), pybamm.Symbol("b")),
                interpolator="cubic spline",
            )

    def test_interpolation(self):
        x = np.linspace(0, 1, 200)
        y = pybamm.StateVector(slice(0, 2))
        # linear
        for interpolator in ["linear", "pchip", "cubic spline"]:
            interp = pybamm.Interpolant(x, 2 * x, y, interpolator=interpolator)
            np.testing.assert_array_almost_equal(
                interp.evaluate(y=np.array([0.397, 1.5]))[:, 0], np.array([0.794, 3])
            )
        # square
        y = pybamm.StateVector(slice(0, 1))
        for interpolator in ["linear", "pchip", "cubic spline"]:
            interp = pybamm.Interpolant(x, x ** 2, y, interpolator=interpolator)
            np.testing.assert_array_almost_equal(
                interp.evaluate(y=np.array([0.397]))[:, 0], np.array([0.397 ** 2])
            )

        # with extrapolation set to False
        for interpolator in ["linear", "pchip", "cubic spline"]:
            interp = pybamm.Interpolant(
                x, x ** 2, y, interpolator=interpolator, extrapolate=False
            )
            np.testing.assert_array_equal(
                interp.evaluate(y=np.array([2]))[:, 0], np.array([np.nan])
            )

    def test_interpolation_1_x_2d_y(self):
        x = np.linspace(0, 1, 200)
        y = np.tile(2 * x, (10, 1)).T
        var = pybamm.StateVector(slice(0, 1))
        # linear
        for interpolator in ["linear", "pchip", "cubic spline"]:
            interp = pybamm.Interpolant(x, y, var, interpolator=interpolator)
            np.testing.assert_array_almost_equal(
                interp.evaluate(y=np.array([0.397])), 0.794 * np.ones((10, 1))
            )

    def test_interpolation_2_x_2d_y(self):
        x = (np.arange(-5.01, 5.01, 0.05), np.arange(-5.01, 5.01, 0.01))
        xx, yy = np.meshgrid(x[0], x[1])
        z = np.sin(xx ** 2 + yy ** 2)
        var1 = pybamm.StateVector(slice(0, 1))
        var2 = pybamm.StateVector(slice(1, 2))
        # linear
        interp = pybamm.Interpolant(x, z, (var1, var2), interpolator="linear")
        np.testing.assert_array_almost_equal(
            interp.evaluate(y=np.array([0, 0])), 0, decimal=3
        )

    def test_name(self):
        a = pybamm.Symbol("a")
        x = np.linspace(0, 1, 200)
        interp = pybamm.Interpolant(x, x, a, "name")
        self.assertEqual(interp.name, "name")
        interp = pybamm.Interpolant(x, x, a)
        self.assertEqual(interp.name, "interpolating_function")

    def test_diff(self):
        x = np.linspace(0, 1, 200)
        y = pybamm.StateVector(slice(0, 2))
        # linear (derivative should be 2)
        # linear interpolator cannot be differentiated
        for interpolator in ["pchip", "cubic spline"]:
            interp_diff = pybamm.Interpolant(
                x, 2 * x, y, interpolator=interpolator
            ).diff(y)
            np.testing.assert_array_almost_equal(
                interp_diff.evaluate(y=np.array([0.397, 1.5]))[:, 0], np.array([2, 2])
            )
        # square (derivative should be 2*x)
        for interpolator in ["pchip", "cubic spline"]:
            interp_diff = pybamm.Interpolant(
                x, x ** 2, y, interpolator=interpolator
            ).diff(y)
            np.testing.assert_array_almost_equal(
                interp_diff.evaluate(y=np.array([0.397, 0.806]))[:, 0],
                np.array([0.794, 1.612]),
                decimal=3,
            )

    def test_processing(self):
        x = np.linspace(0, 1, 200)
        y = pybamm.StateVector(slice(0, 2))
        interp = pybamm.Interpolant(x, 2 * x, y)

        self.assertEqual(interp, interp.new_copy())


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
