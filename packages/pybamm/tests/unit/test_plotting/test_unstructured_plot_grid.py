from types import SimpleNamespace

import casadi
import numpy as np
import pytest

import pybamm
from pybamm.plotting.quick_plot import ax_max, ax_min
from pybamm.plotting.unstructured_plot_grid import plot_grid, quiver_data


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


def _triangle_vector_solution(values=(1.0, 2.0)):
    """Vector field ``(u, u)`` with ``u`` taking ``values`` over time on one
    triangle, so the bounding-box display grid samples points outside the
    domain."""
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
        np.array([0.0, 1.0]), np.asfortranarray([list(values)]), model, {}
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
        grid = plot_grid(solution["u"], n_points=7)
        assert all(len(pts) == 7 for pts in grid.values())

    def test_quiver_data_2d(self):
        solution, (u_val, w_val) = _unstructured_solution(2, 4)
        flux = solution["flux"]
        X, Z, U, W = quiver_data(flux, 0.5, plot_grid(flux))
        for arr in (X, Z, U, W):
            assert arr.shape == (20, 20)
        np.testing.assert_allclose(U[np.isfinite(U)], u_val, rtol=1e-8)
        np.testing.assert_allclose(W[np.isfinite(W)], w_val, rtol=1e-8)


class TestQuickPlotUnstructured:
    def test_2d_scalar_and_vector(self):
        solution, _ = _unstructured_solution(2, 4)
        quick_plot = pybamm.QuickPlot(solution, ["u", "flux"])
        assert list(quick_plot._unstructured_grids[("u",)]) == ["x", "z"]
        quick_plot.plot(0.5)
        image = quick_plot.plots[("u",)][0][1]
        assert image.shape == (200, 200)
        assert np.isfinite(image).mean() > 0.5
        wireframe = quick_plot._wireframes[("u",)]
        quick_plot.slider_update(1.0)
        assert quick_plot.plots[("flux",)][0][0] is not None
        # the field is replaced each frame while the static wireframe is kept
        assert quick_plot._wireframes[("u",)] is wireframe
        assert len(quick_plot.axes[0].collections) == 2
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
        # the default "fixed" limits span the magnitude over all times
        np.testing.assert_allclose(
            quick_plot.variable_limits[("flux",)], (0, np.hypot(u_val, w_val))
        )
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

    def test_zero_vector_field_gets_a_usable_colour_range(self):
        solution = _triangle_vector_solution(values=(0.0, 0.0))
        quick_plot = pybamm.QuickPlot(solution, ["vector"])
        # a degenerate (0, 0) range would leave every arrow the same colour
        assert quick_plot.variable_limits[("vector",)] == (0.0, 1.0)
        quick_plot.plot(0.5)
        norm = quick_plot.plots[("vector",)][0][0].norm
        np.testing.assert_allclose([norm.vmin, norm.vmax], [0.0, 1.0])
        pybamm.close_plots()

    def test_quiver_colour_scale_ignores_samples_outside_domain(self):
        solution = _triangle_vector_solution()
        # |(u, u)| with u = 1 + t peaks at 2 sqrt(2) at t = 1 for fixed limits
        quick_plot = pybamm.QuickPlot(solution, ["vector"])
        quick_plot.plot(0.5)
        norm = quick_plot.plots[("vector",)][0][0].norm
        np.testing.assert_allclose(norm.vmax, 2 * np.sqrt(2))
        # tight limits follow the frame, skipping the NaN samples off the domain
        quick_plot = pybamm.QuickPlot(solution, ["vector"], variable_limits="tight")
        quick_plot.plot(0.5)
        grid = quick_plot._unstructured_grids[("vector",)]
        _, _, U, _ = quiver_data(solution["vector"], 0.5, grid)
        assert np.isnan(U).any()
        norm = quick_plot.plots[("vector",)][0][0].norm
        np.testing.assert_allclose(norm.vmax, 1.5 * np.sqrt(2))
        pybamm.close_plots()

    def test_3d_variables_are_rejected(self):
        # 3D plotting is VTKQuickPlot's job; QuickPlot says where to go instead
        solution, _ = _unstructured_solution(3, 3)
        with pytest.raises(NotImplementedError, match="VTKQuickPlot"):
            pybamm.QuickPlot(solution, ["u"])
        with pytest.raises(NotImplementedError, match="3D vector fields"):
            pybamm.QuickPlot(solution, ["flux"])
        solution = _node_solution()
        with pytest.raises(NotImplementedError, match="VTKQuickPlot"):
            pybamm.QuickPlot(solution, ["node field"])
        # a structured 3D variable is not sent to VTKQuickPlot, which rejects it
        solution._variables["structured"] = SimpleNamespace(
            entries=np.ones(2), dimensions=3, domain=["mesh"]
        )
        with pytest.raises(NotImplementedError, match="3D structured"):
            pybamm.QuickPlot(solution, ["structured"])
