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
        intervals. None means NaN. Default is NaN.

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

    scalar_axes = []
    for name, query in queries.items():
        points = np.asarray(query, dtype=np.float64)
        if points.ndim > 1:
            raise ValueError(
                f"Points to interpolate at in {name!r} must be a scalar or 1D, "
                f"not {points.ndim}D"
            )
        axis = names.index(name)
        result = _interpolate_axis(
            result,
            np.asarray(coords[name], dtype=np.float64),
            np.atleast_1d(points),
            axis,
            name,
            below,
            above,
            extrapolate,
        )
        if points.ndim == 0:
            scalar_axes.append(axis)

    if scalar_axes:
        result = result.squeeze(axis=tuple(scalar_axes))
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


def _interpolate_axis(
    values: npt.NDArray[np.float64],
    grid: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    axis: int,
    name: str,
    below: float,
    above: float,
    extrapolate: bool,
) -> npt.NDArray[np.float64]:
    """Interpolate ``values`` along ``axis``, where it is sampled at ``grid``."""
    if np.any(np.diff(grid) <= 0):
        order = np.argsort(grid, kind="stable")
        grid = grid[order]
        values = np.take(values, order, axis=axis)
        if np.any(np.diff(grid) == 0):
            raise ValueError(
                f"Cannot interpolate in {name!r}: its grid repeats a point"
            )

    if values.ndim == 1 and not extrapolate:
        return np.interp(points, grid, values, left=below, right=above)

    shape = [1] * values.ndim
    shape[axis] = len(points)

    if len(grid) == 1:
        # One point gives a constant, as np.interp gives it
        result = np.repeat(np.take(values, [0], axis=axis), len(points), axis=axis)
        if not extrapolate:
            result = np.where((points < grid[0]).reshape(shape), below, result)
            result = np.where((points > grid[0]).reshape(shape), above, result)
        return np.where(np.isnan(points).reshape(shape), np.nan, result)

    # The interval each point falls in, chosen as np.interp chooses it
    lower = np.clip(np.searchsorted(grid, points, side="right") - 1, 0, len(grid) - 2)
    grid_lower = grid[lower]
    grid_upper = grid[lower + 1]
    values_lower = np.take(values, lower, axis=axis)
    values_upper = np.take(values, lower + 1, axis=axis)
    slope = (values_upper - values_lower) / (grid_upper - grid_lower).reshape(shape)
    result = slope * (points - grid_lower).reshape(shape) + values_lower

    # Grid points take their stored value, as np.interp gives them
    result = np.where((points == grid_lower).reshape(shape), values_lower, result)
    result = np.where((points == grid_upper).reshape(shape), values_upper, result)
    if not extrapolate:
        result = np.where((points < grid[0]).reshape(shape), below, result)
        result = np.where((points > grid[-1]).reshape(shape), above, result)
    return result
