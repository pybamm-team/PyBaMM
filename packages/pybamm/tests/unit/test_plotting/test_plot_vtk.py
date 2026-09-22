from types import SimpleNamespace

import numpy as np
import pytest

import pybamm
from pybamm.plotting.plot_vtk import (
    VTKQuickPlot,
    _build_vtk_grid,
    _compute_scale,
    _make_render_window,
    _mesh_vertices,
    _resolve_scale,
    _set_cell_scalars,
    _set_point_scalars,
    _variable_kind,
    _viridis_lut,
)

vtk = pytest.importorskip("vtk")


def _tetra_mesh():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return pybamm.UnstructuredSubMesh(nodes, np.array([[0, 1, 2, 3]]))


def _cell_solution():
    """One-cell tetrahedral solution; ``shifted`` lives on a second mesh at x in [2, 3]."""
    mesh = _tetra_mesh()
    shifted_mesh = pybamm.UnstructuredSubMesh(
        mesh.vertices + np.array([2.0, 0.0, 0.0]), mesh.elements
    )
    model = pybamm.BaseModel()
    xyz = [pybamm.SpatialVariable(axis, domain="mesh") for axis in "xyz"]
    xyz_shifted = [pybamm.SpatialVariable(axis, domain="shifted") for axis in "xyz"]
    model._geometry = {
        "mesh": {
            var: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)} for var in xyz
        },
        "shifted": {
            var: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
            for var in xyz_shifted
        },
    }

    field = pybamm.StateVector(slice(0, 1), domain="mesh")
    field.mesh = mesh
    shifted = pybamm.StateVector(slice(1, 2), domain="shifted")
    shifted.mesh = shifted_mesh
    model.variables = {"field": field, "shifted": shifted, "scalar": pybamm.t}
    model.update_processed_variables(model.variables)

    t = np.array([0.0, 1.0, 2.0])
    y = np.asfortranarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    return pybamm.Solution(t, y, model, {}), mesh


def _triangle_solution():
    mesh = pybamm.UnstructuredSubMesh(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
        np.array([[0, 1, 2]]),
    )
    model = pybamm.BaseModel()
    x = pybamm.SpatialVariable("x", domain="mesh")
    z = pybamm.SpatialVariable("z", domain="mesh")
    x_line = pybamm.SpatialVariable("x", domain="line")
    model._geometry = {
        "mesh": {
            x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)},
            z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        },
        "line": {x_line: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}},
    }
    field = pybamm.StateVector(slice(0, 1), domain="mesh")
    field.mesh = mesh
    vector = pybamm.VectorField(field, field)
    vector.mesh = mesh
    line = pybamm.StateVector(slice(0, 2), domain="line")
    line.mesh = pybamm.SubMesh1D(np.array([0.0, 0.5, 1.0]), "cartesian")
    model.variables = {"field": field, "vector": vector, "line": line}
    model.update_processed_variables(model.variables)
    solution = pybamm.Solution(
        np.array([0.0, 1.0]), np.asfortranarray([[1.0, 2.0], [3.0, 4.0]]), model, {}
    )
    return solution


def _node_solution():
    # mirrors ScikitFemSubMesh3D, which stores coordinates as ``nodes``
    mesh = SimpleNamespace(
        nodes=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
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

    t = np.array([0.0, 1.0, 2.0])
    y = np.asfortranarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )
    solution = pybamm.Solution(t, y, model, {})
    casadi_field, field, _ = solution._convert_to_casadi(field, {}, y.shape)
    solution._variables["node field"] = pybamm.ProcessedVariableUnstructured(
        "node field", [field], [casadi_field], solution
    )
    return solution, mesh


def _first_actor(renderer):
    actors = renderer.GetActors()
    actors.InitTraversal()
    return actors.GetNextActor()


def _cube_axes(renderer):
    actors = renderer.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextActor()
    while actor is not None and not isinstance(actor, vtk.vtkCubeAxesActor):
        actor = actors.GetNextActor()
    return actor


class TestVTKHelpers:
    @pytest.mark.parametrize(
        ("n_vertices", "cell_type"),
        [
            (3, vtk.VTK_TRIANGLE),
            (4, vtk.VTK_TETRA),
            (8, vtk.VTK_HEXAHEDRON),
        ],
    )
    def test_build_grid_infers_cell_type(self, n_vertices, cell_type):
        nodes = np.column_stack(
            [
                np.arange(n_vertices, dtype=float),
                np.arange(n_vertices, dtype=float) + 1,
                np.arange(n_vertices, dtype=float) + 2,
            ]
        )
        mesh = SimpleNamespace(
            vertices=nodes, elements=np.array([np.arange(n_vertices)])
        )

        grid = _build_vtk_grid(mesh)

        assert grid.GetNumberOfPoints() == n_vertices
        assert grid.GetNumberOfCells() == 1
        assert grid.GetCellType(0) == cell_type
        np.testing.assert_array_equal(
            [grid.GetCell(0).GetPointId(i) for i in range(n_vertices)],
            np.arange(n_vertices),
        )

    def test_build_grid_uses_element_type_and_scales_2d_points(self):
        mesh = SimpleNamespace(
            vertices=np.array([[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]),
            elements=np.array([[0, 1, 2, 3]]),
            element_type="quad",
        )

        grid = _build_vtk_grid(mesh, scale=(2.0, 3.0, 99.0))

        assert grid.GetCellType(0) == vtk.VTK_QUAD
        np.testing.assert_allclose(grid.GetPoint(0), [2.0, 6.0, 0.0])
        np.testing.assert_allclose(grid.GetPoint(2), [6.0, 12.0, 0.0])

    def test_build_grid_connectivity_of_several_cells(self):
        mesh = SimpleNamespace(
            vertices=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            elements=np.array([[0, 1, 2], [0, 2, 3]]),
            element_type="triangle",
        )

        grid = _build_vtk_grid(mesh)

        assert grid.GetNumberOfCells() == 2
        assert grid.GetCellType(1) == vtk.VTK_TRIANGLE
        np.testing.assert_array_equal(
            [grid.GetCell(1).GetPointId(i) for i in range(3)], [0, 2, 3]
        )
        np.testing.assert_allclose(grid.GetPoint(3), [0.0, 1.0, 0.0])

    def test_build_grid_rejects_unknown_connectivity(self):
        mesh = SimpleNamespace(
            vertices=np.zeros((5, 3)), elements=np.array([[0, 1, 2, 3, 4]])
        )

        with pytest.raises(pybamm.GeometryError, match="5 vertices per element"):
            _build_vtk_grid(mesh)

    def test_mesh_vertices_accepts_vertices_or_nodes(self):
        coords = np.array([[0.0, 1.0, 2.0]])
        assert _mesh_vertices(SimpleNamespace(vertices=coords)) is coords
        assert _mesh_vertices(SimpleNamespace(nodes=coords)) is coords

    def test_scale_options(self):
        mesh = SimpleNamespace(nodes=np.array([[0.0, 2.0, 3.0], [4.0, 2.0, 5.0]]))

        np.testing.assert_allclose(_compute_scale(mesh), [1.0, 1.0, 2.0])
        np.testing.assert_allclose(_resolve_scale("auto", mesh), [1.0, 1.0, 2.0])
        assert _resolve_scale(None, mesh) is None
        np.testing.assert_allclose(_resolve_scale((3, 2, 1), mesh), [3, 2, 1])

        zero_mesh = SimpleNamespace(vertices=np.ones((3, 2)))
        np.testing.assert_array_equal(_compute_scale(zero_mesh), [1.0, 1.0])

        with pytest.raises(pybamm.OptionError, match="Unknown scale option"):
            _resolve_scale("big", mesh)
        with pytest.raises(pybamm.OptionError, match="one factor per axis"):
            _resolve_scale(2.0, mesh)
        with pytest.raises(pybamm.OptionError, match="one factor per axis"):
            _resolve_scale((1.0, 2.0), mesh)

    def test_set_and_update_cell_and_point_scalars(self):
        grid = _build_vtk_grid(_tetra_mesh())

        _set_cell_scalars(grid, "cell", [1.25])
        cell_array = grid.GetCellData().GetArray("cell")
        assert grid.GetCellData().GetScalars().GetName() == "cell"
        assert cell_array.GetNumberOfTuples() == 1
        assert cell_array.GetValue(0) == pytest.approx(1.25)

        _set_cell_scalars(grid, "cell", [3.5])
        assert grid.GetCellData().GetArray("cell") is cell_array
        assert cell_array.GetValue(0) == pytest.approx(3.5)

        _set_point_scalars(grid, "point", [0.5, 1.5, 2.5, 3.5])
        point_array = grid.GetPointData().GetArray("point")
        assert grid.GetPointData().GetScalars().GetName() == "point"
        np.testing.assert_allclose(
            [point_array.GetValue(i) for i in range(4)], [0.5, 1.5, 2.5, 3.5]
        )

        _set_point_scalars(grid, "point", [4, 3, 2, 1])
        assert grid.GetPointData().GetArray("point") is point_array
        np.testing.assert_allclose(
            [point_array.GetValue(i) for i in range(4)], [4, 3, 2, 1]
        )

    def test_scalar_length_mismatch_raises(self):
        """A variable must not attach to a grid built from a different mesh.

        Painting a 3-domain variable onto another variable's larger grid
        shifts every value by the leading domains' cell count (e.g.
        electrolyte concentration rendered on current-collector tabs).
        """
        grid = _build_vtk_grid(_tetra_mesh())
        with pytest.raises(pybamm.ShapeError, match="different meshes"):
            _set_cell_scalars(grid, "cell", [1.0, 2.0])
        with pytest.raises(pybamm.ShapeError, match="different meshes"):
            _set_point_scalars(grid, "point", [1.0])

    def test_processed_variable_helpers(self):
        cell_solution, _ = _cell_solution()
        cell_variable = cell_solution["field"]
        scalar_variable = cell_solution["scalar"]
        node_solution, _ = _node_solution()
        node_variable = node_solution["node field"]

        assert _variable_kind(cell_variable) == "cell"
        assert _variable_kind(node_variable) == "node"
        assert _variable_kind(scalar_variable) == "scalar"
        assert _variable_kind(_triangle_solution()["vector"]) is None

    def test_viridis_lookup_table(self):
        lut = _viridis_lut(-2.0, 4.0, n=8)

        assert lut.GetNumberOfTableValues() == 8
        np.testing.assert_allclose(lut.GetRange(), [-2.0, 4.0])
        assert lut.GetTableValue(0)[3] == pytest.approx(1.0)
        assert lut.GetTableValue(7)[3] == pytest.approx(1.0)
        assert lut.GetTableValue(0) != lut.GetTableValue(7)

    def test_make_render_window_offscreen(self):
        window = _make_render_window(off_screen=True)

        assert isinstance(window, vtk.vtkRenderWindow)
        assert window.GetOffScreenRendering() == 1


class TestVTKQuickPlot:
    def test_initialisation_accepts_solution_simulation_and_options(self, monkeypatch):
        solution, _ = _cell_solution()

        # the default variables are the model's plottable quick-plot defaults,
        # never an arbitrary first key such as "Time [s]"
        with pytest.raises(pybamm.OptionError, match="Pass output_variables"):
            VTKQuickPlot(solution)
        # grouped defaults are flattened; line and vector are skipped, not errors
        monkeypatch.setattr(
            pybamm.BaseModel,
            "default_quick_plot_variables",
            property(lambda self: [["line", "vector"], "field"]),
        )
        default_plot = VTKQuickPlot(_triangle_solution())
        assert default_plot.output_variables == ["field"]
        assert default_plot.spatial_panels == [
            ("field", {"plot_type": "3d", "scale": "auto"})
        ]

        simulation = pybamm.Simulation(solution.all_models[0])
        simulation._solution = solution
        plot = VTKQuickPlot(
            simulation,
            "field",
            options={
                "field": [
                    {"plot_type": "3d", "scale": None},
                    {"plot_type": "slice", "z": 0.25},
                ]
            },
            interpolate_time=True,
        )
        assert plot.solution is solution
        assert plot.spatial_names == ["field"]
        assert plot.scalar_names == []
        assert plot.interpolate_time
        assert plot.spatial_panels == [
            ("field", {"plot_type": "3d", "scale": None}),
            (
                "field",
                {"plot_type": "slice", "scale": "auto", "z": 0.25},
            ),
        ]
        assert VTKQuickPlot([solution], "scalar").solution is solution

    def test_dynamic_plot_cell_data_slices_scalar_chart_and_snapped_slider(self):
        solution, _ = _cell_solution()
        plot = VTKQuickPlot(
            solution,
            ["field", "scalar"],
            options={
                "field": [
                    {"plot_type": "3d"},
                    {"plot_type": "slice", "x": 0.4},
                    {"plot_type": "slice", "y": 0.4},
                    {"plot_type": "slice", "z": 0.4},
                ]
            },
        )

        plot.dynamic_plot(show_plot=False)

        assert plot._window.GetWindowName() == "PyBaMM - field, scalar"
        assert plot._window.GetSize() == (1950, 1040)
        assert plot._window.GetRenderers().GetNumberOfItems() == 7
        assert plot._slider.GetEnabled() == 1

        plot._slider.GetRepresentation().SetValue(1.6)
        plot._slider.InvokeEvent("InteractionEvent")

        renderers = plot._window.GetRenderers()
        renderers.InitTraversal()
        field_renderer = renderers.GetNextItem()
        mapped_data = _first_actor(field_renderer).GetMapper().GetInput()
        values = mapped_data.GetPointData().GetArray("field")
        assert values.GetValue(0) == pytest.approx(3.0)

    def test_unwraps_simulation_list_and_rejects_several_solutions(self):
        solution, _ = _cell_solution()
        simulation = pybamm.Simulation(solution.all_models[0])
        simulation._solution = solution
        # BatchStudy.plot hands over a list of simulations
        assert VTKQuickPlot([simulation], "field").solution is solution
        with pytest.raises(pybamm.OptionError, match="single solution"):
            VTKQuickPlot([solution, solution], "field")
        with pytest.raises(TypeError, match="at least 1 solution"):
            VTKQuickPlot([], "field")
        with pytest.raises(pybamm.OptionError, match="at least one output variable"):
            VTKQuickPlot(solution, [])

    def test_nan_cells_do_not_poison_colour_range(self):
        solution, _ = _cell_solution()
        model = solution.all_models[0]
        # the time interpolator spreads a NaN to its neighbouring frames, so
        # leave finite frames at both ends
        t = np.arange(5.0)
        y = np.asfortranarray([[1.0, 2.0, np.nan, 4.0, 5.0], 10.0 * np.arange(1.0, 6)])
        plot = VTKQuickPlot(pybamm.Solution(t, y, model, {}), "field")
        plot.dynamic_plot(show_plot=False)
        renderers = plot._window.GetRenderers()
        renderers.InitTraversal()
        mapper = _first_actor(renderers.GetNextItem()).GetMapper()
        np.testing.assert_allclose(mapper.GetScalarRange(), (1.0, 5.0))
        y[0] = np.nan
        plot = VTKQuickPlot(pybamm.Solution(t, y, model, {}), "field")
        with pytest.raises(pybamm.OptionError, match="no finite values"):
            plot.dynamic_plot(show_plot=False)

    def test_nan_samples_do_not_poison_0d_chart_range(self):
        model = pybamm.BaseModel()
        model.variables = {"state": pybamm.StateVector(slice(0, 1))}
        model.update_processed_variables(model.variables)
        t = np.arange(4.0)
        y = np.asfortranarray([[1.0, np.nan, 3.0, 5.0]])
        plot = VTKQuickPlot(pybamm.Solution(t, y, model, {}), "state")
        plot.dynamic_plot(show_plot=False)

        renderers = plot._window.GetRenderers()
        renderers.InitTraversal()
        props = renderers.GetNextItem().GetViewProps()
        props.InitTraversal()
        chart = props.GetNextProp().GetScene().GetItem(0)
        # the finite range [1, 5] padded by 5 %
        np.testing.assert_allclose(
            [chart.GetAxis(0).GetMinimum(), chart.GetAxis(0).GetMaximum()], [0.8, 5.2]
        )

        y[:] = np.nan
        plot = VTKQuickPlot(pybamm.Solution(t, y, model, {}), "state")
        with pytest.raises(pybamm.OptionError, match="no finite values"):
            plot.dynamic_plot(show_plot=False)

    def test_rejects_vector_field_and_structured_variables(self):
        solution = _triangle_solution()
        with pytest.raises(pybamm.OptionError, match="cannot plot 'vector'"):
            VTKQuickPlot(solution, ["vector"])
        with pytest.raises(pybamm.OptionError, match="cannot plot 'line'"):
            VTKQuickPlot(solution, ["field", "line"])

    def test_slice_and_axes_follow_each_panels_own_mesh(self):
        solution, _ = _cell_solution()
        plot = VTKQuickPlot(
            solution,
            ["field", "shifted"],
            options={"shifted": {"plot_type": "slice", "x": 0.5, "scale": None}},
        )
        plot.dynamic_plot(show_plot=False)

        renderers = plot._window.GetRenderers()
        renderers.InitTraversal()
        renderers.GetNextItem()
        shifted_renderer = renderers.GetNextItem()
        # the cut plane sits at x = 2.5, inside the shifted mesh, so it has cells
        cut = _first_actor(shifted_renderer).GetMapper().GetInput()
        assert cut.GetNumberOfCells() > 0
        np.testing.assert_allclose(cut.GetBounds()[:2], [2.5, 2.5])
        np.testing.assert_allclose(
            _cube_axes(shifted_renderer).GetXAxisRange(), [2.0, 3.0]
        )

    def test_slice_fraction_is_validated_and_kept_inside_the_mesh(self):
        solution, _ = _cell_solution()
        with pytest.raises(pybamm.OptionError, match=r"fraction in \[0, 1\]"):
            VTKQuickPlot(
                solution,
                "field",
                options={"field": {"plot_type": "slice", "x": 1.5}},
            ).dynamic_plot(show_plot=False)

        # a plane exactly on the boundary face would cut nothing
        plot = VTKQuickPlot(
            solution,
            "field",
            options={"field": {"plot_type": "slice", "x": 0.0, "scale": None}},
        )
        plot.dynamic_plot(show_plot=False)
        renderers = plot._window.GetRenderers()
        renderers.InitTraversal()
        cut = _first_actor(renderers.GetNextItem()).GetMapper().GetInput()
        assert cut.GetNumberOfCells() > 0
        np.testing.assert_allclose(cut.GetBounds()[:2], [1e-6, 1e-6], atol=1e-12)

    def test_2d_slice_uses_x_and_z_coordinates(self):
        solution = _triangle_solution()
        plot = VTKQuickPlot(
            solution,
            "field",
            options={"field": {"plot_type": "slice", "z": 0.5, "scale": None}},
        )
        plot.dynamic_plot(show_plot=False)

        renderers = plot._window.GetRenderers()
        renderers.InitTraversal()
        renderer = renderers.GetNextItem()
        cut = _first_actor(renderer).GetMapper().GetInput()
        assert cut.GetNumberOfCells() > 0
        # physical z is drawn on VTK's y axis
        np.testing.assert_allclose(cut.GetBounds()[2:4], [0.5, 0.5])
        cube_axes = _cube_axes(renderer)
        assert cube_axes.GetYTitle() == "z [m]"
        assert cube_axes.GetZAxisVisibility() == 0

        with pytest.raises(pybamm.OptionError, match="coordinates x, z"):
            VTKQuickPlot(
                solution, "field", options={"field": {"plot_type": "slice", "y": 0.5}}
            ).dynamic_plot(show_plot=False)

    def test_dynamic_plot_2d_panels_share_camera(self):
        plot = VTKQuickPlot(
            _triangle_solution(),
            "field",
            options={"field": [{"plot_type": "3d"}, {"plot_type": "3d"}]},
        )
        plot.dynamic_plot(show_plot=False)

        renderers = plot._window.GetRenderers()
        renderers.InitTraversal()
        first = renderers.GetNextItem()
        second = renderers.GetNextItem()
        assert first.GetActiveCamera() is second.GetActiveCamera()
        assert first.GetActiveCamera().GetParallelProjection() == 0

    def test_dynamic_plot_3d_panels_on_different_meshes_get_own_cameras(self):
        solution, _ = _cell_solution()
        plot = VTKQuickPlot(
            solution,
            ["field", "shifted"],
            options={name: {"scale": None} for name in ("field", "shifted")},
        )
        plot.dynamic_plot(show_plot=False)

        renderers = plot._window.GetRenderers()
        renderers.InitTraversal()
        field_camera = renderers.GetNextItem().GetActiveCamera()
        shifted_camera = renderers.GetNextItem().GetActiveCamera()
        assert field_camera is not shifted_camera
        # each camera is framed on its own mesh, x in [0, 1] and x in [2, 3]
        assert field_camera.GetFocalPoint()[0] == pytest.approx(0.5)
        assert shifted_camera.GetFocalPoint()[0] == pytest.approx(2.5)

    def test_dynamic_plot_interpolates_cell_data(self):
        solution, _ = _cell_solution()
        plot = VTKQuickPlot(
            solution,
            "field",
            options={"field": {"scale": None}},
            interpolate_time=True,
        )
        plot.dynamic_plot(show_plot=False)

        plot._slider.GetRepresentation().SetValue(1.25)
        plot._slider.InvokeEvent("InteractionEvent")

        renderers = plot._window.GetRenderers()
        renderers.InitTraversal()
        mapped_data = _first_actor(renderers.GetNextItem()).GetMapper().GetInput()
        values = mapped_data.GetPointData().GetArray("field")
        assert values.GetValue(0) == pytest.approx(2.25)

    def test_dynamic_plot_interpolates_node_data_and_updates_slice(self):
        solution, _ = _node_solution()
        plot = VTKQuickPlot(
            solution,
            "node field",
            options={
                "node field": [
                    {"plot_type": "3d"},
                    {"plot_type": "slice", "x": 0.25},
                ]
            },
            interpolate_time=True,
        )
        plot.dynamic_plot(show_plot=False)

        plot._slider.GetRepresentation().SetValue(1.25)
        plot._slider.InvokeEvent("InteractionEvent")

        renderers = plot._window.GetRenderers()
        renderers.InitTraversal()
        point_data = _first_actor(renderers.GetNextItem()).GetMapper().GetInput()
        values = point_data.GetPointData().GetArray("node field")
        np.testing.assert_allclose(
            [values.GetValue(i) for i in range(4)], [1.25, 2.25, 3.25, 4.25]
        )

    def test_dynamic_plot_node_data_direct_and_slice_pipelines(self):
        solution, _ = _node_solution()
        plot = VTKQuickPlot(
            solution,
            "node field",
            options={
                "node field": [
                    {"plot_type": "3d", "scale": None},
                    {"plot_type": "slice", "z": 0.3, "scale": None},
                ]
            },
        )
        plot.dynamic_plot(show_plot=False)

        renderers = plot._window.GetRenderers()
        renderers.InitTraversal()
        direct_data = _first_actor(renderers.GetNextItem()).GetMapper().GetInput()
        point_values = direct_data.GetPointData().GetArray("node field")
        np.testing.assert_allclose(
            [point_values.GetValue(i) for i in range(4)], [0, 1, 2, 3]
        )

        plot._slider.GetRepresentation().SetValue(2.0)
        plot._slider.InvokeEvent("InteractionEvent")
        np.testing.assert_allclose(
            [point_values.GetValue(i) for i in range(4)], [2, 3, 4, 5]
        )

    def test_dynamic_plot_slice_requires_axis(self):
        solution, _ = _cell_solution()
        plot = VTKQuickPlot(
            solution, "field", options={"field": {"plot_type": "slice"}}
        )

        with pytest.raises(
            pybamm.OptionError, match=r"exactly one of 'x', 'y', or 'z'.*got \[\]"
        ):
            plot.dynamic_plot(show_plot=False)

        plot = VTKQuickPlot(
            solution,
            "field",
            options={"field": {"plot_type": "slice", "y": 0.5, "z": 0.5}},
        )
        with pytest.raises(pybamm.OptionError, match=r"got \['y', 'z'\]"):
            plot.dynamic_plot(show_plot=False)

    def test_options_for_unknown_variables_are_rejected(self):
        solution, _ = _cell_solution()
        with pytest.raises(pybamm.OptionError, match=r"\['Field'\].*not spatial"):
            VTKQuickPlot(solution, "field", options={"Field": {"plot_type": "3d"}})
        # 0D variables have no panel options either
        with pytest.raises(pybamm.OptionError, match=r"\['scalar'\]"):
            VTKQuickPlot(
                solution, ["field", "scalar"], options={"scalar": {"scale": None}}
            )

    def test_empty_panel_option_list_is_rejected(self):
        solution, _ = _cell_solution()
        # an empty list would drop the variable and leave the plot with no panels
        with pytest.raises(pybamm.OptionError, match="options for 'field' are empty"):
            VTKQuickPlot(solution, "field", options={"field": []})

    def test_unknown_panel_options_and_plot_types_are_rejected(self):
        solution, _ = _cell_solution()
        # a typo'd key or plot_type must not fall through to a silent 3D panel
        with pytest.raises(pybamm.OptionError, match=r"\['scael'\] for 'field'"):
            VTKQuickPlot(solution, "field", options={"field": {"scael": None}})
        with pytest.raises(pybamm.OptionError, match=r"plot_type 'Slice'"):
            VTKQuickPlot(
                solution,
                "field",
                options={"field": [{"plot_type": "3d"}, {"plot_type": "Slice"}]},
            )

    def test_save_gif_builds_plot_and_writes_animation(self, tmp_path):
        Image = pytest.importorskip("PIL.Image")
        solution, _ = _cell_solution()
        plot = VTKQuickPlot(solution, "field")
        output = tmp_path / "field.gif"

        plot.save_gif(output, fps=5, n_frames=2, width=160, height=100)
        plot.save_gif(output, fps=5, n_frames=2, width=160, height=100)

        assert output.stat().st_size > 0
        with Image.open(output) as image:
            assert image.size == (160, 100)
            assert image.n_frames == 2
            assert image.info["duration"] == 200


class TestPlotVTKEntryPoints:
    def test_dynamic_plot_vtk_backend(self):
        solution, _ = _cell_solution()
        # positional output_variables, as with the matplotlib backend
        plot = pybamm.dynamic_plot(solution, ["field"], backend="vtk", show_plot=False)
        assert isinstance(plot, pybamm.VTKQuickPlot)
        assert hasattr(plot, "_window")
        plot = pybamm.dynamic_plot(
            solution, output_variables=["field"], backend="vtk", show_plot=False
        )
        assert plot.output_variables == ["field"]

    def test_dynamic_plot_rejects_unknown_backend(self):
        solution, _ = _cell_solution()
        with pytest.raises(pybamm.OptionError, match="Unknown plotting backend"):
            pybamm.dynamic_plot(solution, ["field"], backend="plotly")

    def test_viridis_lut_falls_back_without_matplotlib(self, monkeypatch):
        import sys

        from pybamm.plotting.plot_vtk import _viridis_lut

        monkeypatch.setitem(sys.modules, "matplotlib.cm", None)
        lut = _viridis_lut(0.0, 1.0)
        np.testing.assert_allclose(lut.GetRange(), [0.0, 1.0])
        assert lut.GetNumberOfTableValues() > 0

    def test_make_render_window_on_screen_object(self):
        from pybamm.plotting.plot_vtk import _make_render_window

        # the factory may still return an OSMesa window (VTK_DEFAULT_OPENGL_WINDOW
        # on headless CI), so only the type is asserted
        window = _make_render_window(off_screen=False)
        assert isinstance(window, vtk.vtkRenderWindow)
