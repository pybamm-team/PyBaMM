#
# Tests for the Function classes
#

import re

import casadi
import numpy as np
import pytest

import pybamm


class TestInterpolant:
    def test_errors(self):
        with pytest.raises(ValueError, match=r"x1"):
            pybamm.Interpolant(np.ones(10), np.ones(11), pybamm.Symbol("a"))
        with pytest.raises(ValueError, match=r"x2"):
            pybamm.Interpolant(
                (np.ones(10), np.ones(11)), np.ones((10, 12)), pybamm.Symbol("a")
            )
        with pytest.raises(ValueError, match=r"x1"):
            pybamm.Interpolant(
                (np.ones(11), np.ones(12)), np.ones((10, 12)), pybamm.Symbol("a")
            )
        with pytest.raises(ValueError, match=r"y should"):
            pybamm.Interpolant(
                (np.ones(10), np.ones(11)), np.ones(10), pybamm.Symbol("a")
            )
        with pytest.raises(ValueError, match=r"interpolator 'bla' not recognised"):
            pybamm.Interpolant(
                np.ones(10), np.ones(10), pybamm.Symbol("a"), interpolator="bla"
            )
        with pytest.raises(ValueError, match=r"child should have size 1"):
            pybamm.Interpolant(
                np.ones(10), np.ones((10, 11)), pybamm.StateVector(slice(0, 2))
            )
        with pytest.raises(ValueError, match=r"should equal"):
            pybamm.Interpolant(
                (np.ones(12), np.ones(10)), np.ones((10, 12)), pybamm.Symbol("a")
            )

        with pytest.raises(
            ValueError, match=re.escape("len(x) should equal len(children)")
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
            np.testing.assert_allclose(
                interp.evaluate(y=np.array([0.397, 1.5]))[:, 0],
                np.array([0.794, 3]),
                rtol=1e-7,
                atol=1e-6,
            )
        # square
        y = pybamm.StateVector(slice(0, 1))
        for interpolator in ["linear", "cubic", "pchip"]:
            interp = pybamm.Interpolant(x, x**2, y, interpolator=interpolator)
            np.testing.assert_allclose(
                interp.evaluate(y=np.array([0.397]))[:, 0],
                np.array([0.397**2]),
                rtol=1e-7,
                atol=1e-6,
            )

        # with extrapolation set to False
        for interpolator in ["linear", "cubic", "pchip"]:
            interp = pybamm.Interpolant(
                x, x**2, y, interpolator=interpolator, extrapolate=False
            )
            np.testing.assert_array_equal(
                interp.evaluate(y=np.array([2]))[:, 0], np.array([np.nan])
            )

    def test_interpolation_non_increasing(self):
        x = np.flip(np.linspace(0, 1, 200))
        with pytest.raises(ValueError, match=r"x should be monotonically increasing"):
            pybamm.Interpolant(x, 2 * x, 0.5)

    def test_interpolation_float(self):
        x = np.linspace(0, 1, 200)
        interp = pybamm.Interpolant(x, 2 * x, 0.5)
        assert interp.evaluate() == 1

    def test_interpolation_1_x_2d_y(self):
        x = np.linspace(0, 1, 200)
        y = np.tile(2 * x, (10, 1)).T
        var = pybamm.StateVector(slice(0, 1))
        # linear
        for interpolator in ["linear", "cubic", "pchip"]:
            interp = pybamm.Interpolant(x, y, var, interpolator=interpolator)
            np.testing.assert_allclose(
                interp.evaluate(y=np.array([0.397])),
                0.794 * np.ones((10, 1)),
                rtol=1e-7,
                atol=1e-6,
            )

    def test_interpolation_2_x_2d_y(self):
        x = (np.arange(-5.01, 5.01, 0.05), np.arange(-5.01, 5.01, 0.01))
        xx, yy = np.meshgrid(x[0], x[1], indexing="ij")
        z = np.sin(xx**2 + yy**2)
        var1 = pybamm.StateVector(slice(0, 1))
        var2 = pybamm.StateVector(slice(1, 2))
        # linear
        interp = pybamm.Interpolant(x, z, (var1, var2), interpolator="linear")
        np.testing.assert_allclose(
            interp.evaluate(y=np.array([0, 0])), 0, rtol=1e-4, atol=1e-3
        )
        # cubic
        interp = pybamm.Interpolant(x, z, (var1, var2), interpolator="cubic")
        np.testing.assert_allclose(
            interp.evaluate(y=np.array([0, 0])), 0, rtol=1e-4, atol=1e-3
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
        with pytest.raises(ValueError, match=r"y should be 2-dimensional"):
            interp = pybamm.Interpolant(
                x_in, data_3d, (var1, var2), interpolator="linear"
            )

        # Test raising error if wrong shapes
        with pytest.raises(ValueError, match=r"x1.shape"):
            interp = pybamm.Interpolant(
                x_in, np.zeros((12, 22)), (var1, var2), interpolator="linear"
            )

        with pytest.raises(ValueError, match=r"x2.shape"):
            interp = pybamm.Interpolant(
                x_in, np.zeros((11, 23)), (var1, var2), interpolator="linear"
            )

        # Raise error if not linear
        with pytest.raises(
            ValueError, match=r"interpolator should be 'linear' or 'cubic'"
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
        with pytest.raises(ValueError, match=r"All children must"):
            evaluated_children = [np.array([[1, 1]]), np.array([7])]
            value = interp._function_evaluate(evaluated_children)

        # Test runs when all children are scalars
        evaluated_children = [1, 4]
        value = interp._function_evaluate(evaluated_children)

        # Test that the interpolant shape is the same as the input data shape
        interp = pybamm.Interpolant(x_in, data, (var1, var2), interpolator="linear")

        evaluated_children = [np.array([[1, 1]]), np.array([[7, 7]])]
        value = interp._function_evaluate(evaluated_children)
        assert value.shape == evaluated_children[0].shape

        evaluated_children = [np.array([[1, 1], [1, 1]]), np.array([[7, 7], [7, 7]])]
        value = interp._function_evaluate(evaluated_children)
        assert value.shape == evaluated_children[0].shape

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
        with pytest.raises(ValueError, match=r"y should be 3-dimensional"):
            interp = pybamm.Interpolant(
                x_in, data_4d, (var1, var2, var3), interpolator="linear"
            )

        # Test raising error if wrong shapes
        with pytest.raises(ValueError, match=r"x1.shape"):
            interp = pybamm.Interpolant(
                x_in, np.zeros((12, 22, 33)), (var1, var2, var3), interpolator="linear"
            )

        with pytest.raises(ValueError, match=r"x2.shape"):
            interp = pybamm.Interpolant(
                x_in, np.zeros((11, 23, 33)), (var1, var2, var3), interpolator="linear"
            )

        with pytest.raises(ValueError, match=r"x3.shape"):
            interp = pybamm.Interpolant(
                x_in, np.zeros((11, 22, 34)), (var1, var2, var3), interpolator="linear"
            )

        # Raise error if not linear
        with pytest.raises(
            ValueError, match=r"interpolator should be 'linear' or 'cubic'"
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
        with pytest.raises(ValueError, match=r"All children must"):
            evaluated_children = [np.array([[1, 1]]), np.ones(()) * 4, np.array([[7]])]
            value = interp._function_evaluate(evaluated_children)

        # Test runs when all children are scalsrs
        evaluated_children = [1, 4, 7]
        value = interp._function_evaluate(evaluated_children)

    def test_name(self):
        a = pybamm.Symbol("a")
        x = np.linspace(0, 1, 200)
        interp = pybamm.Interpolant(x, x, a, "name")
        assert interp.name == "name"
        interp = pybamm.Interpolant(x, x, a)
        assert interp.name == "interpolating_function"

    def test_diff(self):
        x = np.linspace(0, 1, 200)
        y = pybamm.StateVector(slice(0, 2))
        # linear (derivative should be 2)
        # linear interpolator cannot be differentiated
        for interpolator in ["cubic", "pchip"]:
            interp_diff = pybamm.Interpolant(
                x, 2 * x, y, interpolator=interpolator
            ).diff(y)
            np.testing.assert_allclose(
                interp_diff.evaluate(y=np.array([0.397, 1.5]))[:, 0],
                np.array([2, 2]),
                rtol=1e-7,
                atol=1e-6,
            )
        # square (derivative should be 2*x)
        for interpolator in ["cubic", "pchip"]:
            interp_diff = pybamm.Interpolant(
                x, x**2, y, interpolator=interpolator
            ).diff(y)
            np.testing.assert_allclose(
                interp_diff.evaluate(y=np.array([0.397, 0.806]))[:, 0],
                np.array([0.794, 1.612]),
                rtol=1e-4,
                atol=1e-3,
            )

        # test 2D interpolation diff fails
        x = (np.arange(-5.01, 5.01, 0.05), np.arange(-5.01, 5.01, 0.01))
        xx, yy = np.meshgrid(x[0], x[1], indexing="ij")
        z = np.sin(xx**2 + yy**2)
        var1 = pybamm.StateVector(slice(0, 1))
        var2 = pybamm.StateVector(slice(1, 2))
        # linear
        interp = pybamm.Interpolant(x, z, (var1, var2), interpolator="linear")
        with pytest.raises(
            NotImplementedError,
            match=r"differentiation not implemented for functions with more than one child",
        ):
            interp.diff(var1)

    @pytest.fixture
    def assert_casadi_matches_evaluate(self):
        def _check(diff_expr, casadi_sym, test_point):
            f = casadi.Function("f", [casadi_sym], [diff_expr.to_casadi(y=casadi_sym)])
            np.testing.assert_allclose(
                np.array(f(test_point)).flatten(),
                diff_expr.evaluate(y=test_point).flatten(),
                rtol=1e-6,
                atol=1e-6,
            )

        return _check

    @pytest.fixture(params=["uniform", "non_uniform"])
    def grid(self, request):
        if request.param == "uniform":
            return np.linspace(0, 1, 200)
        return np.sort(
            np.concatenate([np.linspace(0, 0.5, 50), np.linspace(0.5, 1, 150)[1:]])
        )

    @pytest.mark.parametrize("interpolator", ["cubic", "pchip"])
    def test_diff_to_casadi(self, grid, interpolator, assert_casadi_matches_evaluate):
        # Regression for #5582: the CasADi conversion ignored _num_derivatives
        # and returned the original, un-differentiated function.
        y = pybamm.StateVector(slice(0, 2))
        casadi_y = casadi.MX.sym("y", 2)
        y_test = np.array([0.4, 0.6])
        interp = pybamm.Interpolant(grid, grid**2, y, interpolator=interpolator)

        assert_casadi_matches_evaluate(interp.diff(y), casadi_y, y_test)
        assert_casadi_matches_evaluate(interp.diff(y).diff(y), casadi_y, y_test)

    @pytest.mark.parametrize("interpolator", ["cubic", "pchip"])
    def test_diff_to_casadi_vector_valued(
        self, interpolator, assert_casadi_matches_evaluate
    ):
        x = np.linspace(0, 1, 200)
        data = np.column_stack([x**2, x**3])
        child = pybamm.StateVector(slice(0, 1))
        casadi_child = casadi.MX.sym("y", 1)
        interp = pybamm.Interpolant(x, data, child, interpolator=interpolator)

        assert_casadi_matches_evaluate(
            interp.diff(child), casadi_child, np.array([0.4])
        )

    def test_diff_to_casadi_pchip_third_derivative(
        self, assert_casadi_matches_evaluate
    ):
        # pchip uses a Horner polynomial, not a b-spline, so its (nonzero,
        # piecewise-constant) third derivative converts correctly, unlike cubic.
        x = np.linspace(0, 1, 50)
        y = pybamm.StateVector(slice(0, 1))
        casadi_y = casadi.MX.sym("y", 1)
        third = pybamm.Interpolant(x, x**3, y, interpolator="pchip")
        for _ in range(3):
            third = third.diff(y)
        assert_casadi_matches_evaluate(third, casadi_y, np.array([0.3673]))

    def test_diff_to_casadi_cubic_degree_zero_raises(self):
        # A cubic differentiated three times is a degree-0 spline that CasADi
        # mis-evaluates, so it must raise rather than return wrong values (#5582).
        x = np.linspace(0, 1, 50)
        y = pybamm.StateVector(slice(0, 1))
        casadi_y = casadi.MX.sym("y", 1)
        third = pybamm.Interpolant(x, x**3, y, interpolator="cubic")
        for _ in range(3):
            third = third.diff(y)
        with pytest.raises(NotImplementedError, match="degree-0"):
            third.to_casadi(y=casadi_y)

    def test_processing(self):
        x = np.linspace(0, 1, 200)
        y = pybamm.StateVector(slice(0, 2))
        interp = pybamm.Interpolant(x, 2 * x, y)

        assert interp == interp.create_copy()

    def test_to_from_json(self, mocker):
        x = np.linspace(0, 1, 10)
        y = pybamm.StateVector(slice(0, 2))
        interp = pybamm.Interpolant(x, 2 * x, y)

        expected_json = {
            "name": "interpolating_function",
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
        assert interp.to_json() == expected_json

        expected_json["children"] = [y]
        # check correct re-creation
        assert pybamm.Interpolant._from_json(expected_json) == interp

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

        assert pybamm.Interpolant._from_json(interp2d_json) == interp


class TestInterpolantCasadiEndpoints:
    def test_cubic_matches_scipy_at_the_data_endpoints(self):
        """CasADi's bspline returns 0 outside its knot span and treats the upper
        knot as exclusive, so the last data point used to evaluate to 0."""
        import casadi
        from scipy import interpolate

        x = np.linspace(0.4, 0.998903136, 40)
        y = np.cos(8 * x) + 3
        symbol = casadi.MX.sym("s")
        interpolant = pybamm.Interpolant(
            x, y, pybamm.InputParameter("s"), interpolator="cubic"
        )
        evaluate = casadi.Function(
            "f", [symbol], [interpolant.to_casadi(inputs={"s": symbol})]
        )
        spline = interpolate.make_interp_spline(x, y, k=3)
        for argument in (x[0], x[-1], x[len(x) // 2]):
            assert float(evaluate(argument)) == pytest.approx(
                float(spline(argument)), abs=1e-12
            )
        # and outside the data it holds the endpoint rather than dropping to zero
        for argument in (x[0] - 1e-9, x[-1] + 1e-9):
            assert float(evaluate(argument)) != 0.0


class TestInterpolantMonotonicity:
    """
    ``monotonicity`` is exact: it reads the derivative's extrema off the polynomial
    coefficients of each piece, so an overshoot between the knots cannot be missed.
    """

    def _dense(self, interpolant, region, num_points=2_000_001):
        """The verdict a sweep would give, for comparison against the exact one."""
        grid = np.linspace(*region, num_points)
        difference = np.diff(np.asarray(interpolant.function(grid), float).squeeze())
        if np.all(difference > 0):
            return 1
        return -1 if np.all(difference < 0) else 0

    @pytest.mark.parametrize("interpolator", ["linear", "cubic", "pchip"])
    def test_reports_the_direction(self, interpolator):
        x = np.linspace(0, 1, 11)
        child = pybamm.Variable("v")
        rising = pybamm.Interpolant(x, x**3 + x, child, interpolator=interpolator)
        assert rising.monotonicity() == 1
        falling = pybamm.Interpolant(x, -(x**3) - x, child, interpolator=interpolator)
        assert falling.monotonicity() == -1
        turning = pybamm.Interpolant(
            x, (x - 0.5) ** 2, child, interpolator=interpolator
        )
        assert turning.monotonicity() == 0

    def test_a_constant_interpolant_is_not_strictly_monotonic(self):
        x = np.linspace(0, 1, 5)
        for interpolator in ("linear", "cubic", "pchip"):
            interpolant = pybamm.Interpolant(
                x, np.ones_like(x), pybamm.Variable("v"), interpolator=interpolator
            )
            assert interpolant.monotonicity() == 0

    def test_a_derivative_vanishing_at_a_point_stays_monotonic(self):
        # x^3 is strictly increasing though its derivative is zero at the origin,
        # and a cubic interpolant of it reproduces it exactly
        x = np.linspace(-1, 1, 9)
        interpolant = pybamm.Interpolant(
            x, x**3, pybamm.Variable("v"), interpolator="cubic"
        )
        assert interpolant.monotonicity() == 1

    def test_catches_an_overshoot_a_sweep_misses(self):
        # monotone data whose cubic spline dips by 9e-10; a 1001-point sweep steps
        # over the dip and calls it increasing, the exact answer does not
        x = np.array([0.0, 0.238767, 0.30018, 0.702434, 1.0])
        y = np.array([0.271235, 0.470231, 0.479492, 0.714784, 0.839206])
        interpolant = pybamm.Interpolant(
            x, y, pybamm.Variable("v"), interpolator="cubic"
        )
        assert self._dense(interpolant, (0.0, 1.0), num_points=1001) == 1
        assert self._dense(interpolant, (0.0, 1.0)) == 0
        assert interpolant.monotonicity() == 0
        # pchip is built to preserve the monotonicity of its data, and does
        preserved = pybamm.Interpolant(x, y, pybamm.Variable("v"), interpolator="pchip")
        assert preserved.monotonicity() == 1

    @pytest.mark.parametrize("interpolator", ["linear", "cubic", "pchip"])
    def test_a_region_selects_a_monotonic_stretch(self, interpolator):
        x = np.linspace(0, 1, 21)
        interpolant = pybamm.Interpolant(
            x, (x - 0.5) ** 2, pybamm.Variable("v"), interpolator=interpolator
        )
        assert interpolant.monotonicity() == 0
        assert interpolant.monotonicity((0.0, 0.5)) == -1
        assert interpolant.monotonicity((0.5, 1.0)) == 1

    @pytest.mark.parametrize("interpolator", ["linear", "cubic", "pchip"])
    def test_a_region_reaching_outside_the_data_extrapolates(self, interpolator):
        x = np.linspace(0, 1, 11)
        interpolant = pybamm.Interpolant(
            x, x + 1, pybamm.Variable("v"), interpolator=interpolator
        )
        assert interpolant.monotonicity((-3.0, 4.0)) == 1
        assert self._dense(interpolant, (-3.0, 4.0)) == 1

        blocked = pybamm.Interpolant(
            x, x + 1, pybamm.Variable("v"), interpolator=interpolator, extrapolate=False
        )
        with pytest.raises(ValueError, match="does not extrapolate"):
            blocked.monotonicity((-3.0, 4.0))

    def test_errors(self):
        x = np.linspace(0, 1, 11)
        interpolant = pybamm.Interpolant(
            x, x**2, pybamm.Variable("v"), interpolator="cubic"
        )
        with pytest.raises(ValueError, match="lower < upper"):
            interpolant.monotonicity((0.5, 0.5))

        two_dimensional = pybamm.Interpolant(
            (x, x),
            np.outer(x, x),
            (pybamm.Variable("a"), pybamm.Variable("b")),
            interpolator="linear",
        )
        with pytest.raises(pybamm.ShapeError, match="1D, scalar-valued"):
            two_dimensional.monotonicity()

    @pytest.mark.parametrize("interpolator", ["linear", "cubic", "pchip"])
    def test_never_disagrees_with_a_fine_sweep(self, interpolator):
        """A sweep can miss what the exact answer sees, but never contradict it."""
        rng = np.random.default_rng(20240611)
        child = pybamm.Variable("v")
        for _ in range(60):
            x = np.sort(rng.uniform(0, 1, rng.integers(4, 12)))
            x[0], x[-1] = 0.0, 1.0
            if np.any(np.diff(x) < 1e-3):
                continue
            y = rng.uniform(-1, 1, len(x))
            interpolant = pybamm.Interpolant(x, y, child, interpolator=interpolator)
            lower, upper = sorted(rng.uniform(0, 1, 2))
            if upper - lower < 1e-2:
                lower, upper = 0.0, 1.0
            exact = interpolant.monotonicity((lower, upper))
            swept = self._dense(interpolant, (lower, upper), num_points=200_001)
            assert exact == swept or (exact == 0 and swept != 0)
