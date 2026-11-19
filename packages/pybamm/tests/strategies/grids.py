"""Hypothesis strategies for interpolation on rectilinear grids."""

from __future__ import annotations

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np

_NAMES = ("t", "x", "r")


@st.composite
def grid_interpolation_cases(draw):
    """
    Draw finite values on a grid of 1 to 3 axes, each of 2 to 6 points in a
    random order, with queries of some of its axes and a fill value.

    Returns
    -------
    tuple
        The values, the coordinates by name, the queries by name (scalars or 1D
        arrays, inside and outside the grid, some at its points) and the fill
        value (NaN, a number or ``"extrapolate"``).
    """
    coords = {}
    for name in _NAMES[: draw(st.integers(1, 3))]:
        size = draw(st.integers(2, 6))
        # Spacings of at least 0.5 keep extrapolation from amplifying rounding
        spacings = draw(
            st.lists(st.floats(0.5, 2.0), min_size=size - 1, max_size=size - 1)
        )
        start = draw(st.floats(-10.0, 10.0))
        grid = start + np.concatenate([[0.0], np.cumsum(spacings)])
        coords[name] = grid[draw(st.permutations(range(size)))]

    shape = tuple(grid.size for grid in coords.values())
    values = draw(hnp.arrays(np.float64, shape, elements=st.floats(-100.0, 100.0)))

    queries = {}
    for name in draw(st.lists(st.sampled_from(list(coords)), min_size=1, unique=True)):
        grid = coords[name]
        point = st.one_of(
            st.sampled_from(grid.tolist()),
            st.floats(grid.min() - 1.0, grid.max() + 1.0),
        )
        if draw(st.booleans()):
            queries[name] = draw(point)
        else:
            queries[name] = np.array(draw(st.lists(point, min_size=1, max_size=5)))

    fill_value = draw(
        st.one_of(st.just(np.nan), st.just("extrapolate"), st.floats(-1e3, 1e3))
    )
    return values, coords, queries, fill_value
