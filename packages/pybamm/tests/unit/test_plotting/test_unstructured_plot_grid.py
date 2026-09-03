import casadi
import numpy as np

import pybamm
from pybamm.plotting.unstructured_plot_grid import (
    default_slice_positions,
    midplane_slices,
    plot_grid,
    quiver_data,
)


def _to_casadi(symbol, y):
    t_MX = casadi.MX.sym("t")
    y_MX = casadi.MX.sym("y", y.shape[0])
    inputs_MX = casadi.vertcat()
    return casadi.Function(
        "variable", [t_MX, y_MX, inputs_MX], [symbol.to_casadi(t_MX, y_MX, inputs={})]
    )


def _unstructured_solution(dim, n):
    """Solution on the unit box with scalar ``u = x (1 + t)`` and a constant
    vector field ``flux`` of components ``(2, -3[, 4])``."""
    from pybamm.meshes.unstructured_submesh import UnstructuredMeshGenerator

    domain = "negative electrode"
    x = pybamm.SpatialVariable("x_n", domain=[domain], coord_sys="cartesian")
    if dim == 2:
        z = pybamm.SpatialVariable(
            "z_2d", domain=[domain], coord_sys="cartesian", direction="tb"
        )
        geometry = {domain: {x: {"min": 0, "max": 1}, z: {"min": 0, "max": 1}}}
        var_pts = {x: n, z: n}
        components = (2.0, -3.0)
    else:
        y = pybamm.SpatialVariable("y", domain=[domain], coord_sys="cartesian")
        z = pybamm.SpatialVariable("z", domain=[domain], coord_sys="cartesian")
        geometry = {
            domain: {
                x: {"min": 0, "max": 1},
                y: {"min": 0, "max": 1},
                z: {"min": 0, "max": 1},
            }
        }
        var_pts = {x: n, y: n, z: n}
        components = (2.0, -3.0, 4.0)
    mesh = pybamm.Mesh(geometry, {domain: UnstructuredMeshGenerator()}, var_pts)
    disc = pybamm.Discretisation(mesh, {domain: pybamm.FiniteVolumeUnstructured()})
    var = pybamm.Variable("u", domain=[domain])
    flux = pybamm.VectorField(
        *[pybamm.PrimaryBroadcast(pybamm.Scalar(c), domain) for c in components]
    )
    model = pybamm.BaseModel()
    model.rhs = {var: pybamm.Scalar(0)}
    model.initial_conditions = {var: pybamm.Scalar(0)}
    model.variables = {"u": var, "flux": flux}
    model_disc = disc.process_model(model, inplace=False)
    model_disc._geometry = geometry
    submesh = mesh[domain]
    t_sol = np.array([0.0, 1.0])
    y_sol = submesh.cell_centroids[:, 0][:, np.newaxis] * (1 + t_sol)[np.newaxis, :]
    return pybamm.Solution(t_sol, y_sol, model_disc, {}), components


class TestUnstructuredPlotGrid:
    def test_plot_grid(self):
        solution, _ = _unstructured_solution(2, 4)
        grid = plot_grid(solution["u"])
        assert list(grid) == ["x", "z"]
        assert all(len(pts) == 200 for pts in grid.values())
        np.testing.assert_allclose([grid["x"][0], grid["x"][-1]], [0, 1])
        solution_3d, _ = _unstructured_solution(3, 3)
        grid = plot_grid(solution_3d["u"], n_points=7)
        assert list(grid) == ["x", "y", "z"]
        assert all(len(pts) == 7 for pts in grid.values())
        assert len(plot_grid(solution_3d["u"])["z"]) == 80

    def test_midplane_slices(self):
        solution, _ = _unstructured_solution(3, 3)
        variable = solution["u"]
        grid = plot_grid(variable, n_points=12)
        positions = default_slice_positions(variable)
        np.testing.assert_allclose([positions["y"], positions["z"]], 0.5)
        s1, xx1, yy1, zz1, s2, xx2, yy2, zz2 = midplane_slices(
            variable, 1.0, grid, positions
        )
        for arr in (s1, xx1, yy1, zz1, s2, xx2, yy2, zz2):
            assert arr.shape == (12, 12)
        np.testing.assert_allclose(yy1, 0.5)
        np.testing.assert_allclose(zz2, 0.5)
        # u = 2x at t = 1: linear interpolation between cell centroids is exact
        # between the first and last centroid (x in [1/6, 5/6]); outside the
        # domain the slices are NaN
        assert np.isfinite(s1).mean() > 0.5
        for values, xx in ((s1, xx1), (s2, xx2)):
            interior = (xx > 0.2) & (xx < 0.8)
            np.testing.assert_allclose(values[interior], 2 * xx[interior], atol=1e-8)

    def test_quiver_data_2d(self):
        solution, (u_val, w_val) = _unstructured_solution(2, 4)
        flux = solution["flux"]
        X, Z, U, W = quiver_data(flux, 0.5, plot_grid(flux))
        for arr in (X, Z, U, W):
            assert arr.shape == (20, 20)
        np.testing.assert_allclose(U[np.isfinite(U)], u_val, rtol=1e-8)
        np.testing.assert_allclose(W[np.isfinite(W)], w_val, rtol=1e-8)

    def test_quiver_data_3d(self):
        solution, (u_val, v_val, w_val) = _unstructured_solution(3, 3)
        flux = solution["flux"]
        positions = {"y": 0.4, "z": 0.6}
        data = quiver_data(flux, 0.5, plot_grid(flux), positions, n_points=6)
        X1, Z1, u_xz, w_xz, y_mid, X2, Y2, u_xy, v_xy, z_mid = data
        assert (y_mid, z_mid) == (0.4, 0.6)
        for arr in (X1, Z1, u_xz, w_xz, X2, Y2, u_xy, v_xy):
            assert arr.shape == (6, 6)
        np.testing.assert_allclose(u_xz[np.isfinite(u_xz)], u_val, rtol=1e-8)
        np.testing.assert_allclose(w_xz[np.isfinite(w_xz)], w_val, rtol=1e-8)
        np.testing.assert_allclose(v_xy[np.isfinite(v_xy)], v_val, rtol=1e-8)


class TestQuickPlotUnstructured:
    def test_2d_scalar_and_vector(self):
        solution, _ = _unstructured_solution(2, 4)
        quick_plot = pybamm.QuickPlot(solution, ["u", "flux"])
        assert list(quick_plot._unstructured_grids[("u",)]) == ["x", "z"]
        quick_plot.plot(0.5)
        image = quick_plot.plots[("u",)][0][1]
        assert image.shape == (200, 200)
        assert np.isfinite(image).mean() > 0.5
        quick_plot.slider_update(1.0)
        assert quick_plot.plots[("flux",)][0][0] is not None
        pybamm.close_plots()

    def test_3d_slices_and_slice_sliders(self):
        solution, _ = _unstructured_solution(3, 3)
        quick_plot = pybamm.QuickPlot(solution, ["u", "flux"])
        np.testing.assert_allclose(quick_plot._slice_positions[("u",)]["y"], 0.5)
        quick_plot.dynamic_plot(show_plot=False)
        s1, _ = quick_plot.plots[("u",)][0][0]
        assert s1.shape == (80, 80)
        quick_plot._slice_sliders["y"].set_val(0.25)
        for positions in quick_plot._slice_positions.values():
            np.testing.assert_allclose(positions["y"], 0.25)
        assert quick_plot.plots[("flux",)][0][0] == "quiver_3d"
        pybamm.close_plots()
