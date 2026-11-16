"""Linear interpolation of values sampled on a rectilinear grid."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import numpy.typing as npt


def interpolate_on_grid(
    values: npt.ArrayLike,
    coords: Mapping[str, npt.ArrayLike],
    queries: Mapping[str, Any],
    fill_value: Any = np.nan,
) -> npt.NDArray[np.float64]:
    """
    Linearly interpolate values on a grid, one queried coordinate at a time.

    Parameters
    ----------
    values : array_like
        The values on the grid, whose axes follow the order of ``coords``.
    coords : mapping of str to array_like
        The 1D grid points of each axis, by coordinate name.
    queries : mapping of str to array_like
        The points to interpolate at, by coordinate name. A scalar removes its
        axis from the result and a 1D array replaces the axis with one of its
        length. Axes without a query are kept as they are.
    fill_value : float, (float, float), "extrapolate" or None, optional
        The value outside the grid: one value for both sides, a
        ``(below, above)`` pair, or ``"extrapolate"`` to extend the end
        intervals. None means NaN. A point outside the grid of several queried
        coordinates takes the fill of the last of them. Default is NaN.

    Returns
    -------
    numpy.ndarray
        The interpolated values, as a new array.

    Raises
    ------
    ValueError
        If a query names an unknown coordinate or is not a scalar or 1D, if a
        queried coordinate repeats a grid point, or if ``fill_value`` is invalid.
    """
    result = np.asarray(values, dtype=np.float64)
    if not queries:
        return result.copy()
    names = list(coords)
    unknown = [name for name in queries if name not in coords]
    if unknown:
        raise ValueError(
            f"Cannot interpolate in {unknown}; the coordinates are {names}"
        )
    below, above, extrapolate = _parse_fill_value(fill_value)

    axes = []
    for name, query in queries.items():
        points = np.asarray(query, dtype=np.float64)
        if points.ndim > 1:
            raise ValueError(
                f"Points to interpolate at in {name!r} must be a scalar or 1D, "
                f"not {points.ndim}D"
            )
        axis = names.index(name)
        grid, order = _increasing_grid(coords[name], name)
        axes.append((axis, grid, order, points))

    if result.ndim == 1 and not extrapolate:
        _, grid, order, points = axes[0]
        if order is not None:
            result = result[order]
        return np.asarray(np.interp(points, grid, result, left=below, right=above))

    # Shrink the data as early as possible, so later axes handle less of it
    for axis, grid, order, points in sorted(
        axes, key=lambda item: item[3].size / item[1].size
    ):
        if order is not None:
            result = np.take(result, order, axis=axis)
        result = _interpolate_axis(result, grid, np.atleast_1d(points), axis)

    if not extrapolate:
        for axis, grid, _, points in axes:
            points = np.atleast_1d(points)
            result[_along_axis(result.ndim, axis, points < grid[0])] = below
            result[_along_axis(result.ndim, axis, points > grid[-1])] = above

    scalar_axes = tuple(axis for axis, _, _, points in axes if points.ndim == 0)
    if scalar_axes:
        result = result.squeeze(axis=scalar_axes)
    return result


def _parse_fill_value(fill_value: Any) -> tuple[float, float, bool]:
    """Return the values below and above the grid, and whether to extrapolate."""
    if fill_value is None:
        return np.nan, np.nan, False
    if isinstance(fill_value, str):
        if fill_value == "extrapolate":
            return np.nan, np.nan, True
    elif isinstance(fill_value, Real):
        return float(fill_value), float(fill_value), False
    elif np.ndim(fill_value) == 1 and len(fill_value) == 2:
        below, above = fill_value
        return float(below), float(above), False
    raise ValueError(
        f"fill_value must be a number, a (below, above) pair, 'extrapolate' or "
        f"None, not {fill_value!r}"
    )


def _increasing_grid(
    coord: npt.ArrayLike, name: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp] | None]:
    """Return the grid in increasing order, and the order that sorts it if needed."""
    grid = np.asarray(coord, dtype=np.float64)
    if np.all(np.diff(grid) > 0):
        return grid, None
    order = np.argsort(grid, kind="stable")
    grid = grid[order]
    if np.any(np.diff(grid) == 0):
        raise ValueError(f"Cannot interpolate in {name!r}: its grid repeats a point")
    return grid, order


def _along_axis(ndim: int, axis: int, mask: npt.NDArray[np.bool_]) -> tuple:
    """Index the positions along ``axis`` where ``mask`` is True."""
    index: list[Any] = [slice(None)] * ndim
    index[axis] = np.flatnonzero(mask)
    return tuple(index)


def _interpolate_axis(
    values: npt.NDArray[np.float64],
    grid: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    axis: int,
) -> npt.NDArray[np.float64]:
    """
    Interpolate ``values`` along ``axis``, sampled at the increasing ``grid``,
    extending the end intervals beyond it.
    """
    if len(grid) == 1:
        # One point gives a constant, as np.interp gives it
        result = np.repeat(np.take(values, [0], axis=axis), len(points), axis=axis)
        result[_along_axis(values.ndim, axis, np.isnan(points))] = np.nan
        return result

    # The interval each point falls in, chosen as np.interp chooses it
    lower = np.clip(np.searchsorted(grid, points, side="right") - 1, 0, len(grid) - 2)
    fraction = (points - grid[lower]) / (grid[lower + 1] - grid[lower])
    shape = [1] * values.ndim
    shape[axis] = len(points)
    # In place, so a large result costs two arrays of its size, not several
    values_lower = np.take(values, lower, axis=axis)
    result = np.take(values, lower + 1, axis=axis)
    result -= values_lower
    result *= fraction.reshape(shape)
    result += values_lower

    # Grid points take their stored value, as np.interp gives them
    on_lower = _along_axis(values.ndim, axis, fraction == 0)
    result[on_lower] = values_lower[on_lower]
    on_upper = _along_axis(values.ndim, axis, fraction == 1)
    result[on_upper] = np.take(values, lower[on_upper[axis]] + 1, axis=axis)
    return result
