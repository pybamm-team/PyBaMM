from types import SimpleNamespace

import numpy as np
import pytest

import pybamm
from pybamm.plotting.plot_vtk import (
    VTKQuickPlot,
    _build_vtk_grid,
    _compute_scale,
    _data_at_time,
    _is_unstructured_spatial_variable,
    _resolve_scale,
    _set_cell_scalars,
    _set_point_scalars,
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
    mesh = _tetra_mesh()
    model = pybamm.BaseModel()
    xyz = [pybamm.SpatialVariable(axis, domain="mesh") for axis in "xyz"]
    model._geometry = {
        "mesh": {
            var: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
            for var in xyz
        }
    }

    field = pybamm.StateVector(slice(0, 1), domain="mesh")
    field.mesh = mesh
    model.variables = {"field": field, "scalar": pybamm.t}
    model.update_processed_variables(model.variables)

    t = np.array([0.0, 1.0, 2.0])
    y = np.asfortranarray([[1.0, 2.0, 3.0]])
    return pybamm.Solution(t, y, model, {}), mesh


def _triangle_solution():
    mesh = pybamm.UnstructuredSubMesh(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
        np.array([[0, 1, 2]]),
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
    model.variables = {"field": field}
    model.update_processed_variables(model.variables)
    solution = pybamm.Solution(
        np.array([0.0, 1.0]), np.asfortranarray([[1.0, 2.0]]), model, {}
    )
    return solution


def _node_solution():
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
        "mesh": {
            var: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
            for var in xyz
        }
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
            nodes=nodes, elements=np.array([np.arange(n_vertices)])
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
            nodes=np.array([[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]),
            elements=np.array([[0, 1, 2, 3]]),
            element_type="quad",
        )

        grid = _build_vtk_grid(mesh, scale=(2.0, 3.0, 99.0))

        assert grid.GetCellType(0) == vtk.VTK_QUAD
        np.testing.assert_allclose(grid.GetPoint(0), [2.0, 6.0, 0.0])
        np.testing.assert_allclose(grid.GetPoint(2), [6.0, 12.0, 0.0])

    def test_build_grid_rejects_unknown_connectivity(self):
        mesh = SimpleNamespace(
            nodes=np.zeros((5, 3)), elements=np.array([[0, 1, 2, 3, 4]])
        )

        with pytest.raises(ValueError, match="5 vertices per element"):
            _build_vtk_grid(mesh)

    def test_scale_options(self):
        mesh = SimpleNamespace(
            nodes=np.array([[0.0, 2.0, 3.0], [4.0, 2.0, 5.0]])
        )

        np.testing.assert_allclose(_compute_scale(mesh), [1.0, 1.0, 2.0])
        np.testing.assert_allclose(_resolve_scale("auto", mesh), [1.0, 1.0, 2.0])
        assert _resolve_scale(None, mesh) is None
        np.testing.assert_allclose(_resolve_scale((3, 2, 1), mesh), [3, 2, 1])

        zero_mesh = SimpleNamespace(nodes=np.ones((3, 2)))
        np.testing.assert_array_equal(_compute_scale(zero_mesh), [1.0, 1.0])

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

    def test_processed_variable_helpers(self):
        cell_solution, _ = _cell_solution()
        cell_variable = cell_solution["field"]
        scalar_variable = cell_solution["scalar"]
        node_solution, _ = _node_solution()
        node_variable = node_solution["node field"]

        assert _is_unstructured_spatial_variable(cell_variable)
        assert _is_unstructured_spatial_variable(node_variable)
        assert not _is_unstructured_spatial_variable(scalar_variable)
        np.testing.assert_allclose(_data_at_time(cell_variable, 0.5), [[1.5]])
        assert _data_at_time(scalar_variable, 0.5) == pytest.approx(0.5)

    def test_viridis_lookup_table(self):
        lut = _viridis_lut(-2.0, 4.0, n=8)

        assert lut.GetNumberOfTableValues() == 8
        np.testing.assert_allclose(lut.GetRange(), [-2.0, 4.0])
        assert lut.GetTableValue(0)[3] == pytest.approx(1.0)
        assert lut.GetTableValue(7)[3] == pytest.approx(1.0)
        assert lut.GetTableValue(0) != lut.GetTableValue(7)


class TestVTKQuickPlot:
    def test_initialisation_accepts_solution_simulation_and_options(self):
        solution, mesh = _cell_solution()

        default_plot = VTKQuickPlot(solution)
        assert default_plot.output_variables == ["field"]
        assert default_plot.mesh is mesh
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

        with pytest.raises(ValueError, match="requires one of 'x', 'y', or 'z'"):
            plot.dynamic_plot(show_plot=False)

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
