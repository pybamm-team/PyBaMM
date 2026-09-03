"""Sampling of unstructured-mesh processed variables for plotting.

Unstructured processed variables only interpolate at requested points; the
regular visualisation grid, mid-plane slices and quiver sampling that
:class:`pybamm.QuickPlot` draws are display choices and live here.
"""

from __future__ import annotations

import numpy as np

N_POINTS_2D = 200
N_POINTS_3D = 80
N_QUIVER = 20


def plot_grid(variable, n_points=None):
    """Regular grid over the variable's mesh bounding box.

    Parameters
    ----------
    variable : ProcessedVariableUnstructuredFVM or ProcessedVariableVectorFieldUnstructuredFVM
        The variable to plot.
    n_points : int, optional
        Points per axis; defaults to 200 in 2D and 80 in 3D.

    Returns
    -------
    dict
        One 1D array per axis, keyed ``"x", "z"`` in 2D and ``"x", "y", "z"``
        in 3D, in that order.
    """
    vertices = variable.mesh.vertices
    dimension = variable.mesh.dimension
    if n_points is None:
        n_points = N_POINTS_3D if dimension == 3 else N_POINTS_2D
    names = ("x", "z") if dimension == 2 else ("x", "y", "z")
    return {
        name: np.linspace(vertices[:, k].min(), vertices[:, k].max(), n_points)
        for k, name in enumerate(names)
    }


def default_slice_positions(variable):
    """Mid-plane ``{"y": ..., "z": ...}`` positions of a 3D variable's mesh."""
    vertices = variable.mesh.vertices
    return {
        "y": 0.5 * (vertices[:, 1].min() + vertices[:, 1].max()),
        "z": 0.5 * (vertices[:, 2].min() + vertices[:, 2].max()),
    }


def midplane_slices(variable, t, grid, slice_positions):
    """Two orthogonal slices through a 3D scalar variable at time ``t``.

    Returns ``(s1, xx1, yy1, zz1, s2, xx2, yy2, zz2)``: the x-z plane at
    ``slice_positions["y"]`` followed by the x-y plane at
    ``slice_positions["z"]``, each on the ``grid`` with points outside the
    domain set to NaN.
    """
    x, y, z = grid["x"], grid["y"], grid["z"]
    y_mid, z_mid = slice_positions["y"], slice_positions["z"]
    s1 = variable(t, x=x, y=np.array([y_mid]), z=z).squeeze(axis=1)
    xx1, zz1 = np.meshgrid(x, z, indexing="ij")
    yy1 = np.full_like(xx1, y_mid)
    s2 = variable(t, x=x, y=y, z=np.array([z_mid])).squeeze(axis=2)
    xx2, yy2 = np.meshgrid(x, y, indexing="ij")
    zz2 = np.full_like(xx2, z_mid)
    return s1, xx1, yy1, zz1, s2, xx2, yy2, zz2


def quiver_data(variable, t, grid, slice_positions=None, n_points=N_QUIVER):
    """Vector components on a coarse grid for quiver arrows at time ``t``.

    Returns ``(X, Z, U, W)`` in 2D.  In 3D two mid-plane slices are returned
    as ``(X1, Z1, U_xz, W_xz, y_mid, X2, Y2, U_xy, V_xy, z_mid)``: the x-z
    plane at ``slice_positions["y"]`` followed by the x-y plane at
    ``slice_positions["z"]``.
    """
    x = np.linspace(grid["x"][0], grid["x"][-1], n_points)
    z = np.linspace(grid["z"][0], grid["z"][-1], n_points)
    if variable.dimensions == 2:
        u, w = variable(t, x=x, z=z)
        X, Z = np.meshgrid(x, z, indexing="ij")
        return X, Z, u, w

    y = np.linspace(grid["y"][0], grid["y"][-1], n_points)
    y_mid, z_mid = slice_positions["y"], slice_positions["z"]
    u_xz, _, w_xz = (
        c.squeeze(axis=1) for c in variable(t, x=x, y=np.array([y_mid]), z=z)
    )
    X1, Z1 = np.meshgrid(x, z, indexing="ij")
    u_xy, v_xy, _ = (
        c.squeeze(axis=2) for c in variable(t, x=x, y=y, z=np.array([z_mid]))
    )
    X2, Y2 = np.meshgrid(x, y, indexing="ij")
    return X1, Z1, u_xz, w_xz, y_mid, X2, Y2, u_xy, v_xy, z_mid
