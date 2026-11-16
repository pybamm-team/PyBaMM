import numpy as np
import pytest

from pybamm.solvers.grid_interpolation import interpolate_on_grid


def _bilinear_grid():
    coords = {"x": np.array([0.0, 1.0, 3.0]), "t": np.array([0.0, 2.0, 4.0, 5.0])}
    x, t = np.meshgrid(coords["x"], coords["t"], indexing="ij")
    return 1.0 + 2.0 * x - 0.5 * t + 0.25 * x * t, coords


def _bilinear(x, t):
    return 1.0 + 2.0 * x - 0.5 * t + 0.25 * x * t


class TestInterpolateOnGrid:
    def test_1D_matches_np_interp(self):
        rng = np.random.default_rng(0)
        grid = np.sort(rng.uniform(0, 10, 12))
        values = rng.normal(size=12)
        points = np.concatenate([rng.uniform(-2, 12, 40), grid, [np.nan]])
        rng.shuffle(points)

        result = interpolate_on_grid(values, {"t": grid}, {"t": points})
        expected = np.interp(points, grid, values, left=np.nan, right=np.nan)
        np.testing.assert_array_equal(result, expected)

    def test_multilinear_is_exact_and_keeps_axis_order(self):
        values, coords = _bilinear_grid()
        x_points = np.array([0.5, 2.0, 3.0, 0.0])
        t_points = np.array([1.0, 4.5])

        both = interpolate_on_grid(values, coords, {"t": t_points, "x": x_points})
        assert both.shape == (4, 2)
        np.testing.assert_allclose(
            both, _bilinear(x_points[:, np.newaxis], t_points), rtol=1e-14
        )

        only_t = interpolate_on_grid(values, coords, {"t": t_points})
        assert only_t.shape == (3, 2)
        np.testing.assert_allclose(
            only_t, _bilinear(coords["x"][:, np.newaxis], t_points), rtol=1e-14
        )

    @pytest.mark.parametrize(
        ("x_query", "t_query", "shape"),
        [
            (0.5, 1.0, ()),
            (np.array(0.5), [1.0, 3.0], (2,)),
            ([0.5, 2.0, 2.5], 1.0, (3,)),
            ([0.5], [1.0], (1, 1)),
        ],
    )
    def test_scalar_queries_drop_their_axis(self, x_query, t_query, shape):
        values, coords = _bilinear_grid()
        result = interpolate_on_grid(values, coords, {"x": x_query, "t": t_query})
        assert isinstance(result, np.ndarray)
        assert result.shape == shape
        np.testing.assert_allclose(
            result,
            _bilinear(
                np.asarray(x_query)[..., np.newaxis]
                if np.ndim(t_query)
                else np.asarray(x_query),
                np.asarray(t_query),
            ).reshape(shape),
            rtol=1e-14,
        )

    def test_grid_points_return_stored_values(self):
        values, coords = _bilinear_grid()
        result = interpolate_on_grid(values, coords, dict(coords))
        np.testing.assert_array_equal(result, values)

    def test_no_query_returns_a_copy(self):
        values, coords = _bilinear_grid()
        result = interpolate_on_grid(values, coords, {})
        np.testing.assert_array_equal(result, values)
        assert not np.shares_memory(result, values)

    @pytest.mark.parametrize(
        ("fill_value", "expected"),
        [
            (np.nan, [np.nan, 1.5, np.nan]),
            (None, [np.nan, 1.5, np.nan]),
            (-1.0, [-1.0, 1.5, -1.0]),
            ((-1.0, -2.0), [-1.0, 1.5, -2.0]),
            ("extrapolate", [-2.0, 1.5, 5.0]),
        ],
    )
    def test_fill_value(self, fill_value, expected):
        grid = np.array([0.0, 1.0, 2.0])
        values = 2.0 * grid - 0.5
        result = interpolate_on_grid(
            values, {"t": grid}, {"t": [-0.75, 1.0, 2.75]}, fill_value
        )
        np.testing.assert_array_equal(result, expected)

    def test_fill_comes_from_the_last_queried_coordinate(self):
        values, coords = _bilinear_grid()
        # x = 5 is above its grid and t = -1 below its grid
        fill = (-1.0, -2.0)
        t_last = interpolate_on_grid(values, coords, {"x": 5.0, "t": -1.0}, fill)
        x_last = interpolate_on_grid(values, coords, {"t": -1.0, "x": 5.0}, fill)
        assert t_last == -1.0
        assert x_last == -2.0

    def test_query_order_does_not_change_values(self):
        values, coords = _bilinear_grid()
        x_points = np.linspace(0.0, 3.0, 50)
        t_points = np.array([2.5])
        x_first = interpolate_on_grid(values, coords, {"x": x_points, "t": t_points})
        t_first = interpolate_on_grid(values, coords, {"t": t_points, "x": x_points})
        np.testing.assert_array_equal(x_first, t_first)
        np.testing.assert_allclose(
            x_first, _bilinear(x_points[:, np.newaxis], t_points), rtol=1e-14
        )

    def test_nan_only_reaches_its_own_intervals(self):
        grid = np.arange(5.0)
        values = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
        result = interpolate_on_grid(
            values, {"t": grid}, {"t": [0.5, 1.0, 1.5, 2.5, 3.0, 3.5]}
        )
        np.testing.assert_array_equal(result, [0.5, 1.0, np.nan, np.nan, 3.0, 3.5])

    def test_decreasing_grid(self):
        grid = np.array([3.0, 2.0, 1.0, 0.0])
        result = interpolate_on_grid(np.arange(4.0), {"t": grid}, {"t": [0.5, 2.5]})
        np.testing.assert_array_equal(result, [2.5, 0.5])

    def test_single_point_grid(self):
        result = interpolate_on_grid(
            np.array([[1.0], [2.0]]),
            {"x": np.array([0.0, 1.0]), "t": np.array([3.0])},
            {"t": [2.0, 3.0, 4.0]},
            (-1.0, -2.0),
        )
        np.testing.assert_array_equal(result, [[-1.0, 1.0, -2.0], [-1.0, 2.0, -2.0]])

    def test_errors(self):
        values, coords = _bilinear_grid()
        with pytest.raises(ValueError, match=r"Cannot interpolate in \['r'\]"):
            interpolate_on_grid(values, coords, {"r": 1.0})
        with pytest.raises(ValueError, match=r"'t' must be a scalar or 1D, not 2D"):
            interpolate_on_grid(values, coords, {"t": [[1.0], [2.0]]})
        with pytest.raises(ValueError, match=r"fill_value must be"):
            interpolate_on_grid(values, coords, {"t": 1.0}, "nearest")

        # a repeated grid point only matters along a queried axis
        repeated = {"x": coords["x"], "t": np.array([0.0, 2.0, 2.0, 5.0])}
        with pytest.raises(ValueError, match=r"'t': its grid repeats a point"):
            interpolate_on_grid(values, repeated, {"t": 1.0})
        np.testing.assert_array_equal(
            interpolate_on_grid(values, repeated, {"x": 1.0}), values[1]
        )
