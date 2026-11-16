"""Linear interpolation of values sampled on a rectilinear grid."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

import pybamm


class RectilinearGrid:
    """
    The grid points of each axis of a rectilinear grid, for linear interpolation
    of values sampled on it.

    An axis is checked and sorted the first time it is interpolated along, and the
    result is kept, so repeated interpolation along a long axis stays cheap.

    Parameters
    ----------
    coords : mapping of str to array_like
        The 1D grid points of each axis, by coordinate name, in axis order.
    """

    def __init__(self, coords: Mapping[str, npt.ArrayLike]):
        self.coords = {
            name: np.asarray(points, dtype=np.float64)
            for name, points in coords.items()
        }
        self.shape = tuple(points.size for points in self.coords.values())
        self._increasing: dict[str, tuple[npt.NDArray, npt.NDArray | None]] = {}

    def interpolate(
        self,
        values: npt.ArrayLike,
        queries: Mapping[str, Any],
        fill_value: Any = np.nan,
    ) -> npt.NDArray[np.float64]:
        """
        Linearly interpolate values on the grid, one queried coordinate at a time.

        Parameters
        ----------
        values : array_like
            The values on the grid, whose axes follow the order of the coordinates.
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
        pybamm.ShapeError
            If the values do not match the grid's shape.
        ValueError
            If a query names an unknown coordinate or is not a scalar or 1D, if a
            queried coordinate repeats a grid point, or if ``fill_value`` is
            invalid.
        """
        result = np.asarray(values, dtype=np.float64)
        if result.shape != self.shape:
            raise pybamm.ShapeError(
                f"Values of shape {result.shape} do not match the grid's shape "
                f"{self.shape}"
            )
        names = list(self.coords)
        unknown = [name for name in queries if name not in self.coords]
        if unknown:
            raise ValueError(
                f"Cannot interpolate in {unknown}; the coordinates are {names}"
            )
        below, above, extrapolate = _parse_fill_value(fill_value)

        axes = []
        scalar_axes = []
        for name, query in queries.items():
            points = np.asarray(query, dtype=np.float64)
            if points.ndim > 1:
                raise ValueError(
                    f"Points to interpolate at in {name!r} must be a scalar or 1D, "
                    f"not {points.ndim}D"
                )
            axis = names.index(name)
            if points.ndim == 0:
                scalar_axes.append(axis)
            elif np.array_equal(points, self.coords[name]):
                # The grid's own points need no interpolation
                continue
            grid, order = self._increasing_axis(name)
            axes.append(_AxisQuery(axis, grid, order, np.atleast_1d(points)))

        if not axes:
            result = result.copy()
        elif result.ndim == 1 and not extrapolate:
            query = axes[0]
            if query.order is not None:
                result = result[query.order]
            result = np.interp(
                query.points, query.grid, result, left=below, right=above
            )
        else:
            # Shrink the data as early as possible, so later axes handle less of it
            for query in sorted(
                axes, key=lambda query: query.points.size / query.grid.size
            ):
                if query.order is not None:
                    result = np.take(result, query.order, axis=query.axis)
                result = _interpolate_axis(result, query.grid, query.points, query.axis)
            if not extrapolate:
                for query in axes:
                    outside_below = query.points < query.grid[0]
                    outside_above = query.points > query.grid[-1]
                    result[_along_axis(result.ndim, query.axis, outside_below)] = below
                    result[_along_axis(result.ndim, query.axis, outside_above)] = above

        if scalar_axes:
            result = result.squeeze(axis=tuple(scalar_axes))
        return result

    def _increasing_axis(self, name: str) -> tuple[npt.NDArray, npt.NDArray | None]:
        """
        Return the grid points of an axis in increasing order, and the order that
        sorts them if they were not.
        """
        if name not in self._increasing:
            grid = self.coords[name]
            order = None
            if not np.all(np.diff(grid) > 0):
                order = np.argsort(grid, kind="stable")
                grid = grid[order]
                if np.any(np.diff(grid) == 0):
                    raise ValueError(
                        f"Cannot interpolate in {name!r}: its grid repeats a point"
                    )
            self._increasing[name] = (grid, order)
        return self._increasing[name]


def interpolate_on_grid(
    values: npt.ArrayLike,
    coords: Mapping[str, npt.ArrayLike],
    queries: Mapping[str, Any],
    fill_value: Any = np.nan,
) -> npt.NDArray[np.float64]:
    """
    Linearly interpolate values on the grid ``coords``, as
    :meth:`RectilinearGrid.interpolate` does.

    Parameters
    ----------
    values : array_like
        The values on the grid, whose axes follow the order of ``coords``.
    coords : mapping of str to array_like
        The 1D grid points of each axis, by coordinate name.
    queries : mapping of str to array_like
        The points to interpolate at, by coordinate name.
    fill_value : float, (float, float), "extrapolate" or None, optional
        The value outside the grid. Default is NaN.

    Returns
    -------
    numpy.ndarray
        The interpolated values, as a new array.
    """
    return RectilinearGrid(coords).interpolate(values, queries, fill_value)


class _AxisQuery(NamedTuple):
    """The points to interpolate at along one axis, and that axis's grid."""

    axis: int
    grid: npt.NDArray[np.float64]
    order: npt.NDArray[np.intp] | None
    points: npt.NDArray[np.float64]


def _parse_fill_value(fill_value: Any) -> tuple[float, float, bool]:
    """Return the values below and above the grid, and whether to extrapolate."""
    if fill_value is None:
        return np.nan, np.nan, False
    if isinstance(fill_value, str):
        if fill_value == "extrapolate":
            return np.nan, np.nan, True
    elif np.ndim(fill_value) == 0:
        try:
            value = float(fill_value)
        except (TypeError, ValueError):
            pass
        else:
            return value, value, False
    elif np.ndim(fill_value) == 1 and len(fill_value) == 2:
        below, above = fill_value
        return float(below), float(above), False
    raise ValueError(
        f"fill_value must be a number, a (below, above) pair, 'extrapolate' or "
        f"None, not {fill_value!r}"
    )


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

    # A grid point keeps its stored value, as np.interp gives it. That already
    # holds for finite values; a NaN or inf in the next interval would spoil it
    on_lower = (fraction == 0).reshape(shape)
    if on_lower.any():
        np.copyto(result, values_lower, where=on_lower & ~np.isfinite(result))
    on_upper = _along_axis(values.ndim, axis, fraction == 1)
    result[on_upper] = np.take(values, lower[on_upper[axis]] + 1, axis=axis)
    return result
