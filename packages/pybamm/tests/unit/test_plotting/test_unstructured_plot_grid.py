from types import SimpleNamespace

import casadi
import numpy as np
import pytest

import pybamm
from pybamm.plotting.quick_plot import ax_max, ax_min
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


def _triangle_vector_solution():
    """Vector field ``(u, u)`` with ``u = 1 + t`` on one triangle, so the
    bounding-box display grid samples points outside the domain."""
    mesh = pybamm.UnstructuredSubMesh(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    )
    model = pybamm.BaseModel()
    x = pybamm.SpatialVariable("x", domain="mesh")
    z = pybamm.SpatialVariable("z", domain="mesh")
    model._geometry = {
        "mesh": {
            x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)},
            z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
    }
    field = pybamm.StateVector(slice(0, 1), domain="mesh")
    field.mesh = mesh
    vector = pybamm.VectorField(field, field)
    vector.mesh = mesh
    model.variables = {"vector": vector}
    model.update_processed_variables(model.variables)
    return pybamm.Solution(
        np.array([0.0, 1.0]), np.asfortranarray([[1.0, 2.0]]), model, {}
    )


def _node_solution():
    """Node-centred (scikit-fem style) variable on one tetrahedron."""
    mesh = SimpleNamespace(
        nodes=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        elements=np.array([[0, 1, 2, 3]]),
        dimension=3,
        npts=4,
    )
    model = pybamm.BaseModel()
    xyz = [pybamm.SpatialVariable(axis, domain="mesh") for axis in "xyz"]
    model._geometry = {
        "mesh": {var: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)} for var in xyz}
    }
    field = pybamm.StateVector(slice(0, 4), domain="mesh")
    field.mesh = mesh
    model.variables = {"node field": field}
    model.update_processed_variables(model.variables)
    t = np.array([0.0, 1.0])
    y = np.asfortranarray(np.arange(8.0).reshape(4, 2))
    solution = pybamm.Solution(t, y, model, {})
    casadi_field, field, _ = solution._convert_to_casadi(field, {}, y.shape)
    solution._variables["node field"] = pybamm.ProcessedVariableUnstructured(
        "node field", [field], [casadi_field], solution
    )
    return solution


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
        # the grid spans the closed unit box, so every sample is in the domain
        assert np.isfinite(s1).all() and np.isfinite(s2).all()
        # u = 2x at t = 1 is exact between the first and last centroid
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
        np.testing.assert_allclose([y_mid, z_mid], [0.4, 0.6])
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
        # axes and sliders are in the display unit (um for a unit box in metres)
        scalar_axis, quiver_axis = quick_plot.axes[0], quick_plot.axes[1]
        unit = quick_plot.spatial_unit
        assert scalar_axis.get_xlabel() == quiver_axis.get_xlabel() == f"$x$ [{unit}]"
        np.testing.assert_allclose(scalar_axis.get_xlim(), quiver_axis.get_xlim())
        np.testing.assert_allclose(quick_plot._slice_sliders["y"].val, 0.5e6)
        quick_plot._slice_sliders["y"].set_val(0.25e6)
        for positions in quick_plot._slice_positions.values():
            np.testing.assert_allclose(positions["y"], 0.25)
        assert quick_plot.plots[("flux",)][0][0] == "quiver_3d"
        pybamm.close_plots()

    def test_3d_tight_limits_and_wireframe_guard(self):
        solution, _ = _unstructured_solution(3, 3)
        quick_plot = pybamm.QuickPlot(solution, ["u"], variable_limits="tight")
        quick_plot.plot(0.5)
        quick_plot.slider_update(1.0)
        s1, _ = quick_plot.plots[("u",)][0][0]
        assert np.isfinite(s1).any()
        # the colorbar follows the per-frame range of the slices
        data = solution["u"](1.0)
        norm = quick_plot.colorbars[("u",)].norm
        np.testing.assert_allclose([norm.vmin, norm.vmax], [ax_min(data), ax_max(data)])
        # the 2D wireframe overlay is a no-op on a 3D mesh (returns before drawing)
        assert quick_plot._overlay_mesh_wireframe(None, solution["u"]) is None
        pybamm.close_plots()

    def test_fixed_limits_use_cell_values(self):
        solution, _ = _unstructured_solution(2, 4)
        quick_plot = pybamm.QuickPlot(solution, ["u"])
        # padded extrema of the raw cell values over all times, not of a
        # display-grid interpolation
        cell_values = solution["u"](solution.t)
        np.testing.assert_allclose(
            quick_plot.variable_limits[("u",)],
            (ax_min(cell_values), ax_max(cell_values)),
        )
        pybamm.close_plots()

    def test_vector_field_colour_limits(self):
        solution, (u_val, w_val) = _unstructured_solution(2, 4)
        quick_plot = pybamm.QuickPlot(solution, ["flux"])
        quick_plot.plot(0.5)
        norm = quick_plot.plots[("flux",)][0][0].norm
        np.testing.assert_allclose([norm.vmin, norm.vmax], [0, np.hypot(u_val, w_val)])
        quick_plot = pybamm.QuickPlot(
            solution, ["flux"], variable_limits={"flux": (1.0, 5.0)}
        )
        quick_plot.plot(0.5)
        norm = quick_plot.plots[("flux",)][0][0].norm
        np.testing.assert_allclose([norm.vmin, norm.vmax], [1.0, 5.0])
        pybamm.close_plots()

    def test_quiver_colour_scale_ignores_samples_outside_domain(self):
        solution = _triangle_vector_solution()
        quick_plot = pybamm.QuickPlot(solution, ["vector"])
        quick_plot.plot(0.5)
        grid = quick_plot._unstructured_grids[("vector",)]
        _, _, U, _ = quiver_data(solution["vector"], 0.5, grid)
        assert np.isnan(U).any()
        norm = quick_plot.plots[("vector",)][0][0].norm
        np.testing.assert_allclose(norm.vmax, 1.5 * np.sqrt(2))
        pybamm.close_plots()

    def test_3d_node_centred_variable_points_to_vtk(self):
        solution = _node_solution()
        with pytest.raises(NotImplementedError, match="VTKQuickPlot"):
            pybamm.QuickPlot(solution, ["node field"])
