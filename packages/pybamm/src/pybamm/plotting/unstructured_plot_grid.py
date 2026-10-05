"""Sampling of unstructured-mesh processed variables for plotting.

Unstructured processed variables only interpolate at requested points; the
regular visualisation grid and quiver sampling that :class:`pybamm.QuickPlot`
draws are display choices and live here.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pybamm

Grid = dict[str, npt.NDArray[np.float64]]

N_POINTS = 200
N_QUIVER = 20


def plot_grid(
    variable: pybamm.ProcessedVariableUnstructuredFVM
    | pybamm.ProcessedVariableVectorFieldUnstructuredFVM,
    n_points: int = N_POINTS,
) -> Grid:
    """Regular grid over a 2D variable's mesh bounding box.

    Parameters
    ----------
    variable : ProcessedVariableUnstructuredFVM or ProcessedVariableVectorFieldUnstructuredFVM
        The 2D variable to plot.
    n_points : int, optional
        Points per axis. Default is 200.

    Returns
    -------
    dict
        One 1D array per axis, keyed ``"x"`` then ``"z"``.
    """
    vertices = variable.mesh.vertices
    return {
        name: np.linspace(vertices[:, k].min(), vertices[:, k].max(), n_points)
        for k, name in enumerate(("x", "z"))
    }


def quiver_data(
    variable: pybamm.ProcessedVariableVectorFieldUnstructuredFVM,
    t: float,
    grid: Grid,
    n_points: int = N_QUIVER,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Vector components of a 2D variable on a coarse grid at time ``t``.

    Returns ``(X, Z, U, W)``: the meshgrid of sample points and the x and z
    components there.
    """
    x = np.linspace(grid["x"][0], grid["x"][-1], n_points)
    z = np.linspace(grid["z"][0], grid["z"][-1], n_points)
    u, w = variable(t, x=x, z=z)
    X, Z = np.meshgrid(x, z, indexing="ij")
    return X, Z, u, w
