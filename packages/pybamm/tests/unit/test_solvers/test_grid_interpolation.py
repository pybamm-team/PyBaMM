import warnings

import numpy as np
import pytest
from hypothesis import given
from scipy.interpolate import RegularGridInterpolator

import pybamm
from pybamm.solvers.grid_interpolation import RectilinearGrid
from tests.strategies.grids import grid_interpolation_cases


def interpolate_on_grid(values, coords, queries, fill_value=np.nan):
    return RectilinearGrid(coords).interpolate(values, queries, fill_value)


def _bilinear_grid():
    coords = {"x": np.array([0.0, 1.0, 3.0]), "t": np.array([0.0, 2.0, 4.0, 5.0])}
    x, t = np.meshgrid(coords["x"], coords["t"], indexing="ij")
    return 1.0 + 2.0 * x - 0.5 * t + 0.25 * x * t, coords


def _bilinear(x, t):
    return 1.0 + 2.0 * x - 0.5 * t + 0.25 * x * t


class TestRectilinearGrid:
    def test_1D_matches_np_interp(self):
        rng = np.random.default_rng(0)
        grid = np.sort(rng.uniform(0, 10, 12))
        values = rng.normal(size=12)
        points = np.concatenate([rng.uniform(-2, 12, 40), grid, [np.nan]])
        rng.shuffle(points)

        result = interpolate_on_grid(values, {"t": grid}, {"t": points})
        expected = np.interp(points, grid, values, left=np.nan, right=np.nan)
        np.testing.assert_array_equal(result, expected)

    @given(grid_interpolation_cases())
    def test_matches_scipy_regular_grid_interpolator(self, case):
        values, coords, queries, fill_value = case
        result = interpolate_on_grid(values, coords, queries, fill_value)

        # scipy needs each axis increasing; it extrapolates and fills come after
        grids = []
        for axis, points in enumerate(coords.values()):
            order = np.argsort(points)
            grids.append(points[order])
            values = np.take(values, order, axis=axis)
        interpolator = RegularGridInterpolator(
            grids, values, bounds_error=False, fill_value=None
        )
        # An axis without a query keeps its own order
        axis_points = [
            np.atleast_1d(queries.get(name, coords[name])) for name in coords
        ]
        expected = interpolator(np.stack(np.meshgrid(*axis_points, indexing="ij"), -1))
        for axis, (name, grid) in enumerate(zip(coords, grids, strict=True)):
            if name in queries and not isinstance(fill_value, str):
                points = axis_points[axis]
                index = [slice(None)] * expected.ndim
                index[axis] = (points < grid[0]) | (points > grid[-1])
                expected[tuple(index)] = fill_value
        scalar_axes = tuple(
            axis
            for axis, name in enumerate(coords)
            if name in queries and np.ndim(queries[name]) == 0
        )
        expected = expected.squeeze(axis=scalar_axes)

        assert result.shape == expected.shape
        np.testing.assert_allclose(
            result, expected, rtol=1e-12, atol=1e-12 * np.abs(values).max()
        )

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
        expected = _bilinear(np.reshape(x_query, (-1, 1)), np.reshape(t_query, (1, -1)))
        np.testing.assert_allclose(result, expected.reshape(shape), rtol=1e-14)

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
            (np.array(-1.0), [-1.0, 1.5, -1.0]),
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

    def test_nan_beside_a_grid_point_keeps_its_stored_value(self):
        # With a second axis, this takes the N-D route rather than np.interp
        values = np.array([[0.0, 1.0, np.nan, 3.0], [0.0, 1.0, 2.0, np.inf]])
        coords = {"x": np.array([0.0, 1.0]), "t": np.arange(4.0)}
        result = interpolate_on_grid(values, coords, {"t": [1.0, 1.5, 2.0, 3.0]})
        np.testing.assert_array_equal(
            result, [[1.0, np.nan, np.nan, 3.0], [1.0, 1.5, 2.0, np.inf]]
        )

    def test_inf_beside_a_grid_point_does_not_warn(self):
        # With "extrapolate", this takes the N-D route rather than np.interp
        values = np.array([0.0, np.inf, 2.0])
        # The suite ignores RuntimeWarning, so make it an error here
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = interpolate_on_grid(
                values, {"t": np.arange(3.0)}, {"t": [0.0, 1.0, 1.5]}, "extrapolate"
            )
        # Grid points keep their stored values; between inf and 2 is NaN
        np.testing.assert_array_equal(result, [0.0, np.inf, np.nan])

    def test_decreasing_grid(self):
        grid = np.array([3.0, 2.0, 1.0, 0.0])
        result = interpolate_on_grid(np.arange(4.0), {"t": grid}, {"t": [0.5, 2.5]})
        np.testing.assert_array_equal(result, [2.5, 0.5])

    @pytest.mark.parametrize("flipped_axis", [0, 1])
    def test_decreasing_grid_on_several_axes(self, flipped_axis):
        values, coords = _bilinear_grid()
        name = list(coords)[flipped_axis]
        flipped_coords = {**coords, name: coords[name][::-1]}
        flipped_values = np.flip(values, axis=flipped_axis)
        x_points, t_points = np.array([0.5, 2.0, 3.0]), np.array([1.0, 4.5])

        result = interpolate_on_grid(
            flipped_values, flipped_coords, {"x": x_points, "t": t_points}
        )
        np.testing.assert_allclose(
            result, _bilinear(x_points[:, np.newaxis], t_points), rtol=1e-14
        )

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
        with pytest.raises(
            pybamm.ShapeError, match=r"shape \(3, 3\) do not match .* \(3, 4\)"
        ):
            interpolate_on_grid(values[:, :3], coords, {"t": 1.0})
        with pytest.raises(ValueError, match=r"Cannot interpolate in \['r'\]"):
            interpolate_on_grid(values, coords, {"r": 1.0})
        with pytest.raises(ValueError, match=r"'t' must be a scalar or 1D, not 2D"):
            interpolate_on_grid(values, coords, {"t": [[1.0], [2.0]]})
        for fill_value in ["nearest", np.array("nearest"), object(), (1.0, 2.0, 3.0)]:
            with pytest.raises(ValueError, match=r"fill_value must be"):
                interpolate_on_grid(values, coords, {"t": 1.0}, fill_value)

        # a repeated grid point only matters along a queried axis
        repeated = {"x": coords["x"], "t": np.array([0.0, 2.0, 2.0, 5.0])}
        with pytest.raises(ValueError, match=r"'t': its grid repeats a point"):
            interpolate_on_grid(values, repeated, {"t": 1.0})
        np.testing.assert_array_equal(
            interpolate_on_grid(values, repeated, {"x": 1.0}), values[1]
        )
        # and the grid's own points need no interpolation
        np.testing.assert_array_equal(
            interpolate_on_grid(values, repeated, {"t": repeated["t"]}), values
        )

    def test_grid_checks_each_axis_once(self):
        values, coords = _bilinear_grid()
        grid = RectilinearGrid(coords)
        first = grid.interpolate(values, {"t": [1.0, 2.5]})
        checked = grid._increasing["t"]
        np.testing.assert_array_equal(
            grid.interpolate(values, {"t": [1.0, 2.5]}), first
        )
        assert grid._increasing["t"] is checked
        assert "x" not in grid._increasing
