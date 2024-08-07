#
# Tests for the Function classes
#

import pybamm

import unittest
import unittest.mock as mock
import numpy as np


class TestInterpolant(unittest.TestCase):
    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "x1"):
            pybamm.Interpolant(np.ones(10), np.ones(11), pybamm.Symbol("a"))
        with self.assertRaisesRegex(ValueError, "x2"):
            pybamm.Interpolant(
                (np.ones(10), np.ones(11)), np.ones((10, 12)), pybamm.Symbol("a")
            )
        with self.assertRaisesRegex(ValueError, "x1"):
            pybamm.Interpolant(
                (np.ones(11), np.ones(12)), np.ones((10, 12)), pybamm.Symbol("a")
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

        with self.assertRaisesRegex(
            ValueError, "len\\(x\\) should equal len\\(children\\)"
        ):
            pybamm.Interpolant(
                (np.ones(10), np.ones(12)), np.ones((10, 12)), pybamm.Symbol("a")
            )

    def test_interpolation(self):
        x = np.linspace(0, 1, 200)
        y = pybamm.StateVector(slice(0, 2))
        # linear
        for interpolator in ["linear", "cubic", "pchip"]:
            interp = pybamm.Interpolant(x, 2 * x, y, interpolator=interpolator)
            np.testing.assert_array_almost_equal(
                interp.evaluate(y=np.array([0.397, 1.5]))[:, 0], np.array([0.794, 3])
            )
        # square
        y = pybamm.StateVector(slice(0, 1))
        for interpolator in ["linear", "cubic", "pchip"]:
            interp = pybamm.Interpolant(x, x**2, y, interpolator=interpolator)
            np.testing.assert_array_almost_equal(
                interp.evaluate(y=np.array([0.397]))[:, 0], np.array([0.397**2])
            )

        # with extrapolation set to False
        for interpolator in ["linear", "cubic", "pchip"]:
            interp = pybamm.Interpolant(
                x, x**2, y, interpolator=interpolator, extrapolate=False
            )
            np.testing.assert_array_equal(
                interp.evaluate(y=np.array([2]))[:, 0], np.array([np.nan])
            )

    def test_interpolation_float(self):
        x = np.linspace(0, 1, 200)
        interp = pybamm.Interpolant(x, 2 * x, 0.5)
        self.assertEqual(interp.evaluate(), 1)

    def test_interpolation_1_x_2d_y(self):
        x = np.linspace(0, 1, 200)
        y = np.tile(2 * x, (10, 1)).T
        var = pybamm.StateVector(slice(0, 1))
        # linear
        for interpolator in ["linear", "cubic", "pchip"]:
            interp = pybamm.Interpolant(x, y, var, interpolator=interpolator)
            np.testing.assert_array_almost_equal(
                interp.evaluate(y=np.array([0.397])), 0.794 * np.ones((10, 1))
            )

    def test_interpolation_2_x_2d_y(self):
        x = (np.arange(-5.01, 5.01, 0.05), np.arange(-5.01, 5.01, 0.01))
        xx, yy = np.meshgrid(x[0], x[1], indexing="ij")
        z = np.sin(xx**2 + yy**2)
        var1 = pybamm.StateVector(slice(0, 1))
        var2 = pybamm.StateVector(slice(1, 2))
        # linear
        interp = pybamm.Interpolant(x, z, (var1, var2), interpolator="linear")
        np.testing.assert_array_almost_equal(
            interp.evaluate(y=np.array([0, 0])), 0, decimal=3
        )
        # cubic
        interp = pybamm.Interpolant(x, z, (var1, var2), interpolator="cubic")
        np.testing.assert_array_almost_equal(
            interp.evaluate(y=np.array([0, 0])), 0, decimal=3
        )

    def test_interpolation_2_x(self):
        def f(x, y):
            return 2 * x**3 + 3 * y**2

        x = np.linspace(1, 4, 11)
        y = np.linspace(4, 7, 22)
        xg, yg = np.meshgrid(x, y, indexing="ij", sparse=True)
        data = f(xg, yg)

        var1 = pybamm.StateVector(slice(0, 1))
        var2 = pybamm.StateVector(slice(1, 2))

        x_in = (x, y)
        interp = pybamm.Interpolant(x_in, data, (var1, var2), interpolator="linear")

        value = interp.evaluate(y=np.array([1, 5]))
        np.testing.assert_equal(value, f(1, 5))

        value = interp.evaluate(y=np.array([x[1], y[1]]))
        np.testing.assert_equal(value, f(x[1], y[1]))

        value = interp.evaluate(y=np.array([[1, 1, x[1]], [5, 4, y[1]]]))
        np.testing.assert_array_equal(
            value, np.array([[f(1, 5), f(1, 4), f(x[1], y[1])]])
        )

        # check also works for cubic
        interp = pybamm.Interpolant(x_in, data, (var1, var2), interpolator="cubic")
        value = interp.evaluate(y=np.array([1, 5]))
        np.testing.assert_almost_equal(value, f(1, 5), decimal=3)

        # Test raising error if data is not 2D
        data_3d = np.zeros((11, 22, 33))
        with self.assertRaisesRegex(ValueError, "y should be two-dimensional"):
            interp = pybamm.Interpolant(
                x_in, data_3d, (var1, var2), interpolator="linear"
            )

        # Test raising error if wrong shapes
        with self.assertRaisesRegex(ValueError, "x1.shape"):
            interp = pybamm.Interpolant(
                x_in, np.zeros((12, 22)), (var1, var2), interpolator="linear"
            )

        with self.assertRaisesRegex(ValueError, "x2.shape"):
            interp = pybamm.Interpolant(
                x_in, np.zeros((11, 23)), (var1, var2), interpolator="linear"
            )

        # Raise error if not linear
        with self.assertRaisesRegex(
            ValueError, "interpolator should be 'linear' or 'cubic'"
        ):
            interp = pybamm.Interpolant(x_in, data, (var1, var2), interpolator="pchip")

        # Check returns nan if extrapolate set to False
        interp = pybamm.Interpolant(
            x_in, data, (var1, var2), interpolator="linear", extrapolate=False
        )
        value = interp.evaluate(y=np.array([0, 0, 0]))
        np.testing.assert_equal(value, np.nan)

        # Check testing for shape works (i.e. using nans)
        interp = pybamm.Interpolant(x_in, data, (var1, var2), interpolator="cubic")
        interp.test_shape()

        # test with inconsistent children shapes
        # (this can occur is one child is a scaler and the others
        # are variables)
        evaluated_children = [np.array([[1]]), 4]
        value = interp._function_evaluate(evaluated_children)

        evaluated_children = [np.array([[1]]), np.ones(()) * 4]
        value = interp._function_evaluate(evaluated_children)

        # Test evaluation fails with different child shapes
        with self.assertRaisesRegex(ValueError, "All children must"):
            evaluated_children = [np.array([[1, 1]]), np.array([7])]
            value = interp._function_evaluate(evaluated_children)

        # Test runs when all children are scalars
        evaluated_children = [1, 4]
        value = interp._function_evaluate(evaluated_children)

        # Test that the interpolant shape is the same as the input data shape
        interp = pybamm.Interpolant(x_in, data, (var1, var2), interpolator="linear")

        evaluated_children = [np.array([[1, 1]]), np.array([[7, 7]])]
        value = interp._function_evaluate(evaluated_children)
        self.assertEqual(value.shape, evaluated_children[0].shape)

        evaluated_children = [np.array([[1, 1], [1, 1]]), np.array([[7, 7], [7, 7]])]
        value = interp._function_evaluate(evaluated_children)
        self.assertEqual(value.shape, evaluated_children[0].shape)

    def test_interpolation_3_x(self):
        def f(x, y, z):
            return 2 * x**3 + 3 * y**2 - z

        x = np.linspace(1, 4, 11)
        y = np.linspace(4, 7, 22)
        z = np.linspace(7, 9, 33)
        xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)
        data = f(xg, yg, zg)

        var1 = pybamm.StateVector(slice(0, 1))
        var2 = pybamm.StateVector(slice(1, 2))
        var3 = pybamm.StateVector(slice(2, 3))

        x_in = (x, y, z)
        interp = pybamm.Interpolant(
            x_in, data, (var1, var2, var3), interpolator="linear"
        )

        value = interp.evaluate(y=np.array([1, 5, 8]))
        np.testing.assert_equal(value, f(1, 5, 8))

        value = interp.evaluate(y=np.array([[1, 1, 1], [5, 4, 4], [8, 7, 7]]))
        np.testing.assert_array_equal(
            value, np.array([[f(1, 5, 8), f(1, 4, 7), f(1, 4, 7)]])
        )

        # check also works for cubic
        interp = pybamm.Interpolant(
            x_in, data, (var1, var2, var3), interpolator="cubic"
        )
        value = interp.evaluate(y=np.array([1, 5, 8]))
        np.testing.assert_almost_equal(value, f(1, 5, 8), decimal=3)

        # Test raising error if data is not 3D
        data_4d = np.zeros((11, 22, 33, 5))
        with self.assertRaisesRegex(ValueError, "y should be three-dimensional"):
            interp = pybamm.Interpolant(
                x_in, data_4d, (var1, var2, var3), interpolator="linear"
            )

        # Test raising error if wrong shapes
        with self.assertRaisesRegex(ValueError, "x1.shape"):
            interp = pybamm.Interpolant(
                x_in, np.zeros((12, 22, 33)), (var1, var2, var3), interpolator="linear"
            )

        with self.assertRaisesRegex(ValueError, "x2.shape"):
            interp = pybamm.Interpolant(
                x_in, np.zeros((11, 23, 33)), (var1, var2, var3), interpolator="linear"
            )

        with self.assertRaisesRegex(ValueError, "x3.shape"):
            interp = pybamm.Interpolant(
                x_in, np.zeros((11, 22, 34)), (var1, var2, var3), interpolator="linear"
            )

        # Raise error if not linear
        with self.assertRaisesRegex(
            ValueError, "interpolator should be 'linear' or 'cubic'"
        ):
            interp = pybamm.Interpolant(
                x_in, data, (var1, var2, var3), interpolator="pchip"
            )

        # Check returns nan if extrapolate set to False
        interp = pybamm.Interpolant(
            x_in, data, (var1, var2, var3), interpolator="linear", extrapolate=False
        )
        value = interp.evaluate(y=np.array([0, 0, 0]))
        np.testing.assert_equal(value, np.nan)

        # Check testing for shape works (i.e. using nans)
        interp = pybamm.Interpolant(
            x_in, data, (var1, var2, var3), interpolator="cubic"
        )
        interp.test_shape()

        # test with inconsistent children shapes
        # (this can occur is one child is a scaler and the others
        # are vaiables)
        evaluated_children = [np.array([[1]]), 4, np.array([[7]])]
        value = interp._function_evaluate(evaluated_children)

        evaluated_children = [np.array([[1]]), np.ones(()) * 4, np.array([[7]])]
        value = interp._function_evaluate(evaluated_children)

        # Test evaluation fails with different child shapes
        with self.assertRaisesRegex(ValueError, "All children must"):
            evaluated_children = [np.array([[1, 1]]), np.ones(()) * 4, np.array([[7]])]
            value = interp._function_evaluate(evaluated_children)

        # Test runs when all children are scalsrs
        evaluated_children = [1, 4, 7]
        value = interp._function_evaluate(evaluated_children)

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
        for interpolator in ["cubic", "pchip"]:
            interp_diff = pybamm.Interpolant(
                x, 2 * x, y, interpolator=interpolator
            ).diff(y)
            np.testing.assert_array_almost_equal(
                interp_diff.evaluate(y=np.array([0.397, 1.5]))[:, 0], np.array([2, 2])
            )
        # square (derivative should be 2*x)
        for interpolator in ["cubic", "pchip"]:
            interp_diff = pybamm.Interpolant(
                x, x**2, y, interpolator=interpolator
            ).diff(y)
            np.testing.assert_array_almost_equal(
                interp_diff.evaluate(y=np.array([0.397, 0.806]))[:, 0],
                np.array([0.794, 1.612]),
                decimal=3,
            )

        # test 2D interpolation diff fails
        x = (np.arange(-5.01, 5.01, 0.05), np.arange(-5.01, 5.01, 0.01))
        xx, yy = np.meshgrid(x[0], x[1], indexing="ij")
        z = np.sin(xx**2 + yy**2)
        var1 = pybamm.StateVector(slice(0, 1))
        var2 = pybamm.StateVector(slice(1, 2))
        # linear
        interp = pybamm.Interpolant(x, z, (var1, var2), interpolator="linear")
        with self.assertRaisesRegex(
            NotImplementedError,
            "differentiation not implemented for functions with more than one child",
        ):
            interp.diff(var1)

    def test_processing(self):
        x = np.linspace(0, 1, 200)
        y = pybamm.StateVector(slice(0, 2))
        interp = pybamm.Interpolant(x, 2 * x, y)

        self.assertEqual(interp, interp.create_copy())

    def test_to_from_json(self):
        x = np.linspace(0, 1, 10)
        y = pybamm.StateVector(slice(0, 2))
        interp = pybamm.Interpolant(x, 2 * x, y)

        expected_json = {
            "name": "interpolating_function",
            "id": mock.ANY,
            "x": [
                [
                    0.0,
                    0.1111111111111111,
                    0.2222222222222222,
                    0.3333333333333333,
                    0.4444444444444444,
                    0.5555555555555556,
                    0.6666666666666666,
                    0.7777777777777777,
                    0.8888888888888888,
                    1.0,
                ]
            ],
            "y": [
                0.0,
                0.2222222222222222,
                0.4444444444444444,
                0.6666666666666666,
                0.8888888888888888,
                1.1111111111111112,
                1.3333333333333333,
                1.5555555555555554,
                1.7777777777777777,
                2.0,
            ],
            "interpolator": "linear",
            "extrapolate": True,
            "_num_derivatives": 0,
        }

        # check correct writing to json
        self.assertEqual(interp.to_json(), expected_json)

        expected_json["children"] = [y]
        # check correct re-creation
        self.assertEqual(pybamm.Interpolant._from_json(expected_json), interp)

        # test to_from_json for 2d x & y
        x = (np.arange(-5.01, 5.01, 0.05), np.arange(-5.01, 5.01, 0.01))
        xx, yy = np.meshgrid(x[0], x[1], indexing="ij")
        z = np.sin(xx**2 + yy**2)
        var1 = pybamm.StateVector(slice(0, 1))
        var2 = pybamm.StateVector(slice(1, 2))
        # linear
        interp = pybamm.Interpolant(x, z, (var1, var2), interpolator="linear")

        interp2d_json = interp.to_json()
        interp2d_json["children"] = (var1, var2)

        self.assertEqual(pybamm.Interpolant._from_json(interp2d_json), interp)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
