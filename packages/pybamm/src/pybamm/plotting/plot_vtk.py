"""
VTK-based interactive visualization for unstructured mesh solutions.

Provides :class:`VTKQuickPlot`, a drop-in alternative to the matplotlib-based
:class:`QuickPlot` for 2D and 3D unstructured mesh data (cell-centered FVM
and node-centered FEM).

Also supports 0D (time-series) variables rendered as VTK line charts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import pybamm

_VTK_CELL_TYPE = {
    "triangle": 5,  # VTK_TRIANGLE
    "quad": 9,  # VTK_QUAD
    "tetrahedron": 10,  # VTK_TETRA
    "hexahedron": 12,  # VTK_HEXAHEDRON
}
_PLOT_TYPES = frozenset({"3d", "slice"})
_PANEL_OPTION_KEYS = frozenset({"plot_type", "scale", "x", "y", "z"})


def _mesh_vertices(mesh):
    """Vertex coordinates of an unstructured mesh.

    Finite-volume meshes store them as ``vertices``; the scikit-fem 3D mesh
    that backs node-centred variables stores them as ``nodes``.
    """
    return mesh.vertices if hasattr(mesh, "vertices") else mesh.nodes


def _build_vtk_grid(mesh, scale=None):
    """Build a ``vtkUnstructuredGrid`` from an unstructured mesh."""
    vtk = pybamm.import_optional_dependency("vtk")
    numpy_support = pybamm.import_optional_dependency("vtk.util.numpy_support")

    nodes = np.asarray(_mesh_vertices(mesh), dtype=float)
    if scale is not None:
        nodes = nodes * np.asarray(scale)[: nodes.shape[1]]
    if nodes.shape[1] == 2:
        nodes = np.column_stack([nodes, np.zeros(len(nodes))])

    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(nodes), deep=True))

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(pts)

    if hasattr(mesh, "element_type"):
        element_key = mesh.element_type
    else:
        nverts = mesh.elements.shape[1]
        if nverts == 4:
            element_key = "tetrahedron"
        elif nverts == 8:
            element_key = "hexahedron"
        elif nverts == 3:
            element_key = "triangle"
        else:
            raise pybamm.GeometryError(
                "Unable to infer VTK cell type from mesh connectivity with "
                f"{nverts} vertices per element"
            )

    cell_type = _VTK_CELL_TYPE[element_key]
    elements = np.asarray(mesh.elements)
    n_cells, n_verts = elements.shape
    id_type = np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_support.numpy_to_vtkIdTypeArray(
            np.arange(0, (n_cells + 1) * n_verts, n_verts, dtype=id_type), deep=True
        ),
        numpy_support.numpy_to_vtkIdTypeArray(
            np.ascontiguousarray(elements.ravel(), dtype=id_type), deep=True
        ),
    )
    grid.SetCells(cell_type, cells)

    return grid


def _compute_scale(mesh):
    """Per-axis scale factors that normalise coordinate spans to the largest."""
    nodes = _mesh_vertices(mesh)
    spans = np.array(
        [nodes[:, d].max() - nodes[:, d].min() for d in range(nodes.shape[1])]
    )
    max_span = spans.max()
    if max_span == 0:
        return np.ones(nodes.shape[1])
    return max_span / np.where(spans > 0, spans, max_span)


def _resolve_scale(scale_opt, mesh):
    """Turn a scale option into a concrete array or None."""
    if scale_opt is None:
        return None
    if isinstance(scale_opt, str):
        if scale_opt == "auto":
            return _compute_scale(mesh)
        raise pybamm.OptionError(
            f"Unknown scale option {scale_opt!r}: use 'auto', None or one factor "
            "per axis."
        )
    scale = np.asarray(scale_opt, dtype=float)
    dimension = _mesh_vertices(mesh).shape[1]
    if scale.ndim != 1 or len(scale) < dimension:
        raise pybamm.OptionError(
            f"The scale option needs one factor per axis ({dimension} for this "
            f"mesh), got {scale_opt!r}."
        )
    return scale


def _set_scalars(attribute_data, expected, kind, name, values):
    """Set (or update) a named float scalar array on cell or point data."""
    numpy_support = pybamm.import_optional_dependency("vtk.util.numpy_support")

    values = np.ascontiguousarray(values, dtype=np.float32).ravel()
    if len(values) != expected:
        raise pybamm.ShapeError(
            f"Cannot attach {len(values)} {kind} values for {name!r} to a grid "
            f"with {expected} {kind}s: the variable and the grid describe "
            "different meshes."
        )
    arr = attribute_data.GetArray(name)
    if arr is None:
        arr = numpy_support.numpy_to_vtk(values, deep=True)
        arr.SetName(name)
        attribute_data.AddArray(arr)
        attribute_data.SetActiveScalars(name)
    else:
        # one vectorised copy into VTK's buffer instead of a per-value loop
        numpy_support.vtk_to_numpy(arr)[:] = values
        arr.Modified()


def _set_cell_scalars(grid, name, values):
    """Set (or update) a cell scalar array on a VTK grid."""
    _set_scalars(grid.GetCellData(), grid.GetNumberOfCells(), "cell", name, values)
    grid.Modified()


def _set_point_scalars(grid, name, values):
    """Set (or update) a point scalar array on a VTK grid."""
    _set_scalars(grid.GetPointData(), grid.GetNumberOfPoints(), "point", name, values)
    grid.Modified()


def _variable_kind(pv):
    """Classify a processed variable for VTK plotting.

    Returns ``"cell"`` for cell-centred unstructured data, ``"node"`` for
    node-centred unstructured data, ``"scalar"`` for a 0D time series and
    ``None`` for anything VTKQuickPlot cannot draw.
    """
    if isinstance(pv, pybamm.ProcessedVariableUnstructuredFVM):
        return "cell"
    if isinstance(pv, pybamm.ProcessedVariableUnstructured):
        return "node"
    if getattr(pv, "dimensions", None) == 0:
        return "scalar"
    return None


def _finite_range(name, values):
    """Min and max of the finite samples; NaNs must not set the range."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise pybamm.OptionError(f"'{name}' has no finite values to plot")
    return float(finite.min()), float(finite.max())


def _viridis_lut(vmin, vmax, n=256):
    """Build a VTK lookup table using the matplotlib viridis colormap."""
    vtk = pybamm.import_optional_dependency("vtk")

    try:
        _cmap = pybamm.import_optional_dependency("matplotlib.cm", "viridis")
    except ModuleNotFoundError:
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0)
        lut.SetRange(vmin, vmax)
        lut.Build()
        return lut

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.SetRange(vmin, vmax)
    for i in range(n):
        r, g, b, a = _cmap(i / (n - 1))
        lut.SetTableValue(i, r, g, b, a)
    lut.Build()
    return lut


def _make_render_window(off_screen=False):
    """Create a VTK render window.

    VTK's object factory picks the window class, so a headless host selects
    OSMesa with ``VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow`` while a
    desktop keeps its native OpenGL window.
    """
    vtk = pybamm.import_optional_dependency("vtk")

    window = vtk.vtkRenderWindow()
    if off_screen:
        window.SetOffScreenRendering(1)
    return window


class VTKQuickPlot:
    """Interactive VTK visualization for unstructured mesh solutions.

    Supports spatial (unstructured 2D/3D) and 0D (time-series) variables.

    Parameters
    ----------
    solutions : :class:`pybamm.Solution` or :class:`pybamm.Simulation`
        The solution to plot; a single-element list is also accepted.
    output_variables : str or list of str, optional
        Variables to plot. Defaults to the model's default quick-plot
        variables that are 0D or live on an unstructured mesh.
    options : dict, optional
        Per-variable options keyed by variable name.  Each value is a dict
        that may contain:

        - ``"plot_type"``: ``"3d"`` (default) or ``"slice"``
        - ``"x"`` / ``"y"`` / ``"z"``: float in [0, 1] giving the slice
          position as a fraction of the axis range (required when
          ``plot_type`` is ``"slice"``)
        - ``"scale"``: ``"auto"`` (default), ``None``, or ``(sx, sy, sz)``

        A variable's value may also be a **list** of such dicts, in which
        case one panel is created per entry::

            options={"T": [
                {"plot_type": "3d"},
                {"plot_type": "slice", "x": 0.5},
            ]}
    interpolate_time : bool, optional
        Evaluate the variables at the exact slider time instead of snapping
        to the nearest stored solution time. Default is False.
    """

    def __init__(
        self,
        solutions: pybamm.Solution
        | pybamm.Simulation
        | list[pybamm.Solution | pybamm.Simulation],
        output_variables: str | list[str] | None = None,
        options: dict[str, dict[str, Any] | list[dict[str, Any]]] | None = None,
        interpolate_time: bool = False,
    ):
        solutions = pybamm.QuickPlot.preprocess_solutions(solutions)
        if len(solutions) != 1:
            raise pybamm.OptionError(
                f"VTKQuickPlot plots a single solution, but {len(solutions)} were "
                "given. Use pybamm.QuickPlot to compare solutions."
            )
        self.solution = solutions[0]

        if output_variables is None:
            output_variables = self._default_output_variables()
        if isinstance(output_variables, str):
            output_variables = [output_variables]
        if len(output_variables) == 0:
            raise pybamm.OptionError(
                "VTKQuickPlot needs at least one output variable to plot."
            )

        self.spatial_names = []
        self.spatial_vars = []
        self.spatial_is_cell_data = []
        self.scalar_names = []
        self.scalar_vars = []

        for name in output_variables:
            pv = self.solution[name]
            kind = _variable_kind(pv)
            if kind in ("cell", "node"):
                self.spatial_names.append(name)
                self.spatial_vars.append(pv)
                self.spatial_is_cell_data.append(kind == "cell")
            elif kind == "scalar":
                self.scalar_names.append(name)
                self.scalar_vars.append(pv)
            else:
                raise pybamm.OptionError(
                    f"VTKQuickPlot cannot plot '{name}': only scalar variables on "
                    "unstructured meshes and 0D time series are supported. Use "
                    "pybamm.QuickPlot for structured-mesh and vector-field "
                    "variables."
                )

        self.output_variables = output_variables
        self.t_pts = self.solution.t
        self.interpolate_time = interpolate_time

        _defaults = {"plot_type": "3d", "scale": "auto"}
        raw_opts = options or {}
        unknown = sorted(set(raw_opts) - set(self.spatial_names))
        if unknown:
            raise pybamm.OptionError(
                f"Options were given for {unknown}, which are not spatial output "
                f"variables of this plot ({self.spatial_names})."
            )

        self.spatial_panels = []
        for name in self.spatial_names:
            var_opt = raw_opts.get(name, _defaults)
            if isinstance(var_opt, dict):
                opt_list = [var_opt]
            else:
                opt_list = list(var_opt)
            if not opt_list:
                raise pybamm.OptionError(
                    f"The options for '{name}' are empty, so it would get no panel; "
                    "pass one option dict per panel, or drop it from "
                    "output_variables."
                )
            for single_opt in opt_list:
                unknown_keys = sorted(set(single_opt) - _PANEL_OPTION_KEYS)
                if unknown_keys:
                    raise pybamm.OptionError(
                        f"Unknown option(s) {unknown_keys} for '{name}'; the panel "
                        f"options are {sorted(_PANEL_OPTION_KEYS)}."
                    )
                merged = dict(_defaults)
                merged.update(single_opt)
                if merged["plot_type"] not in _PLOT_TYPES:
                    raise pybamm.OptionError(
                        f"Unknown plot_type {merged['plot_type']!r} for '{name}'; "
                        f"use one of {sorted(_PLOT_TYPES)}."
                    )
                self.spatial_panels.append((name, merged))

    def _default_output_variables(self) -> list[str]:
        """The model's default quick-plot variables that VTK can draw."""
        defaults = self.solution.all_models[0].default_quick_plot_variables or []
        # a default may group several names onto one QuickPlot axis; VTK draws
        # one panel per variable, so the groups are flattened
        names = [
            name
            for entry in defaults
            for name in ([entry] if isinstance(entry, str) else entry)
        ]
        plottable = [
            name for name in names if _variable_kind(self.solution[name]) is not None
        ]
        if not plottable:
            raise pybamm.OptionError(
                "VTKQuickPlot has no default variables for this model: none of "
                f"its default quick-plot variables {names} are 0D or on "
                "an unstructured mesh. Pass output_variables explicitly."
            )
        return plottable

    def dynamic_plot(self, show_plot: bool = True) -> None:
        """Launch an interactive VTK window with a time slider."""
        vtk = pybamm.import_optional_dependency("vtk")
        numpy_support = pybamm.import_optional_dependency("vtk.util.numpy_support")

        n_spatial = len(self.spatial_panels)
        n_scalar = len(self.scalar_names)
        n_panels = n_spatial + n_scalar

        spatial_data = {}
        spatial_mins = {}
        spatial_maxs = {}
        for name, pv in zip(self.spatial_names, self.spatial_vars, strict=True):
            data = np.asarray(pv(self.t_pts), dtype=float)
            spatial_data[name] = data
            spatial_mins[name], spatial_maxs[name] = _finite_range(name, data)

        scalar_data = {}
        scalar_ranges = {}
        for name, pv in zip(self.scalar_names, self.scalar_vars, strict=True):
            vals = np.asarray(pv(self.t_pts), dtype=float).ravel()
            scalar_data[name] = vals
            scalar_ranges[name] = _finite_range(name, vals)

        slider_h = 0.08
        panel_top = 1.0
        panel_bot = slider_h

        n_cols = int(np.ceil(np.sqrt(n_panels)))
        n_rows = int(np.ceil(n_panels / n_cols))
        panel_height = (panel_top - panel_bot) / n_rows

        def viewport(index):
            """Normalised (xmin, ymin, xmax, ymax) of panel ``index``, row-major."""
            row, col = divmod(index, n_cols)
            return (
                col / n_cols,
                panel_top - (row + 1) * panel_height,
                (col + 1) / n_cols,
                panel_top - row * panel_height,
            )

        window = _make_render_window(off_screen=not show_plot)
        window.SetSize(650 * n_cols, 520 * n_rows)
        window.SetWindowName("PyBaMM - " + ", ".join(self.output_variables))

        all_renderers = []
        spatial_grids = []
        cell_to_point_filters = []
        cutters = []
        chart_views = []
        time_markers = []

        panel_idx = 0

        # 3d panels whose scaled grids share bounds (same mesh) share one camera
        shared_cameras = {}
        spatial_renderers = []
        panel_names = []
        is_cell_data_by_name = {
            name: is_cell
            for name, is_cell in zip(
                self.spatial_names, self.spatial_is_cell_data, strict=True
            )
        }

        pv_by_name = dict(zip(self.spatial_names, self.spatial_vars, strict=True))
        for name, opts in self.spatial_panels:
            plot_type = opts.get("plot_type", "3d")
            # each variable is drawn on its own mesh: painting a 3-domain variable
            # onto a 5-domain grid would shift every value by the leading cells
            panel_mesh = pv_by_name[name].mesh
            panel_nodes = _mesh_vertices(panel_mesh)
            dim = panel_nodes.shape[1]
            # column k of the vertex array is drawn on VTK axis k; 2D meshes
            # store their in-plane coordinates as (x, z)
            axis_names = ("x", "z") if dim == 2 else ("x", "y", "z")
            axis_columns = {axis: k for k, axis in enumerate(axis_names)}
            var_scale = _resolve_scale(opts.get("scale", "auto"), panel_mesh)
            is_cell_data = is_cell_data_by_name[name]
            panel_names.append(name)

            g = _build_vtk_grid(panel_mesh, scale=var_scale)
            if is_cell_data:
                _set_cell_scalars(g, name, spatial_data[name][:, 0])
            else:
                _set_point_scalars(g, name, spatial_data[name][:, 0])
            spatial_grids.append(g)

            cell_to_point = None
            if is_cell_data:
                cell_to_point = vtk.vtkCellDataToPointData()
                cell_to_point.SetInputData(g)
                cell_to_point.Update()
            cell_to_point_filters.append(cell_to_point)

            pipeline_source = (
                cell_to_point.GetOutputPort() if cell_to_point is not None else g
            )
            cutter = None
            if plot_type == "slice":
                axes_given = [ak for ak in ("x", "y", "z") if ak in opts]
                if len(axes_given) != 1:
                    raise pybamm.OptionError(
                        f"plot_type='slice' for '{name}' requires exactly one of "
                        f"'x', 'y', or 'z' specifying the slice fraction, got "
                        f"{axes_given}."
                    )
                axis_key = axes_given[0]
                if axis_key not in axis_columns:
                    raise pybamm.OptionError(
                        f"Cannot slice '{name}' along '{axis_key}': its "
                        f"{dim}D mesh has coordinates {', '.join(axis_names)}."
                    )
                axis_idx = axis_columns[axis_key]
                frac = float(opts[axis_key])
                if not 0.0 <= frac <= 1.0:
                    raise pybamm.OptionError(
                        f"The slice position for '{name}' must be a fraction in "
                        f"[0, 1] of the {axis_key} range, got {frac}."
                    )
                lo = float(panel_nodes[:, axis_idx].min())
                hi = float(panel_nodes[:, axis_idx].max())
                # a plane exactly on a boundary face cuts nothing: stay just inside
                margin = 1e-6 * (hi - lo)
                phys_val = min(max(lo + frac * (hi - lo), lo + margin), hi - margin)
                scaled_val = (
                    phys_val * var_scale[axis_idx]
                    if var_scale is not None
                    else phys_val
                )

                plane = vtk.vtkPlane()
                origin = [0.0, 0.0, 0.0]
                origin[axis_idx] = scaled_val
                plane.SetOrigin(origin)
                normal = [0.0, 0.0, 0.0]
                normal[axis_idx] = 1.0
                plane.SetNormal(normal)

                cutter = vtk.vtkCutter()
                cutter.SetCutFunction(plane)
                if cell_to_point is not None:
                    cutter.SetInputConnection(pipeline_source)
                else:
                    cutter.SetInputData(pipeline_source)
                cutter.Update()

                mapper_source = cutter.GetOutputPort()
            else:
                if cell_to_point is not None:
                    mapper_source = pipeline_source
                else:
                    mapper_source = None

            cutters.append(cutter)

            lut = _viridis_lut(spatial_mins[name], spatial_maxs[name])

            mapper = vtk.vtkDataSetMapper()
            if mapper_source is not None:
                mapper.SetInputConnection(mapper_source)
            else:
                mapper.SetInputData(g)
            mapper.SetScalarRange(spatial_mins[name], spatial_maxs[name])
            mapper.SetScalarModeToUsePointData()
            mapper.SelectColorArray(name)
            mapper.SetLookupTable(lut)
            mapper.InterpolateScalarsBeforeMappingOn()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if plot_type == "slice":
                actor.GetProperty().EdgeVisibilityOff()
            else:
                actor.GetProperty().EdgeVisibilityOn()
                actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.2)
                actor.GetProperty().SetLineWidth(0.3)

            sb = vtk.vtkScalarBarActor()
            sb.SetLookupTable(lut)
            sb.SetTitle("")
            sb.SetNumberOfLabels(5)
            sb.SetWidth(0.2)
            sb.SetHeight(0.5)
            sb.SetPosition(0.79, 0.25)
            sb.GetLabelTextProperty().SetFontSize(22)
            sb.GetLabelTextProperty().SetColor(0, 0, 0)
            sb.SetUnconstrainedFontSize(True)
            # 4 significant figures without a width spec: "1265" and "303.2"
            # rather than a clipped "1.27e+" or a dangling "303."
            sb.SetLabelFormat("%.4g")

            title_actor = vtk.vtkTextActor()
            title_actor.SetInput(name)
            title_actor.GetTextProperty().SetFontSize(30)
            title_actor.GetTextProperty().SetColor(0, 0, 0)
            title_actor.GetTextProperty().SetBold(True)
            title_actor.GetTextProperty().SetJustificationToCentered()
            title_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            title_actor.SetPosition(0.5, 0.92)

            ren = vtk.vtkRenderer()
            ren.AddActor(actor)
            ren.AddViewProp(sb)
            ren.AddViewProp(title_actor)
            ren.SetBackground(1, 1, 1)

            ren.SetViewport(*viewport(panel_idx))

            if plot_type == "slice":
                # Use the cutter output bounds so axes align with
                # the visible slice geometry, not the full 3D grid.
                axes_bounds = list(cutter.GetOutput().GetBounds())
            else:
                axes_bounds = list(g.GetBounds())

            cube_axes = vtk.vtkCubeAxesActor()
            cube_axes.SetBounds(axes_bounds)
            cube_axes.SetUseAxisOrigin(False)
            cube_axes.SetFlyModeToOuterEdges()
            if plot_type == "slice":
                cube_axes.SetTickLocationToInside()
            cube_axes.SetScreenSize(8.0)
            cube_axes.SetLabelOffset(8)
            cube_axes.SetTitleOffset([16, 16])
            # print coordinates as they are, without a "(x10^-6)" factor
            cube_axes.SetLabelScaling(False, 0, 0, 0)

            orig_ranges = [
                (float(panel_nodes[:, d].min()), float(panel_nodes[:, d].max()))
                for d in range(dim)
            ]
            if dim >= 1:
                cube_axes.SetXAxisRange(*orig_ranges[0])
            if dim >= 2:
                cube_axes.SetYAxisRange(*orig_ranges[1])
            if dim >= 3:
                cube_axes.SetZAxisRange(*orig_ranges[2])

            for ax_id in range(3):
                tp = cube_axes.GetTitleTextProperty(ax_id)
                tp.SetFontSize(22)
                tp.SetColor(0.15, 0.15, 0.15)
                tp.SetBold(True)
                lp = cube_axes.GetLabelTextProperty(ax_id)
                lp.SetFontSize(17)
                lp.SetColor(0.25, 0.25, 0.25)
            cube_axes.SetXTitle(f"{axis_names[0]} [m]")
            cube_axes.SetYTitle(f"{axis_names[1]} [m]")
            if dim == 3:
                cube_axes.SetZTitle("z [m]")
            else:
                cube_axes.ZAxisVisibilityOff()
            cube_axes.SetXLabelFormat("%.3g")
            cube_axes.SetYLabelFormat("%.3g")
            cube_axes.SetZLabelFormat("%.3g")
            cube_axes.XAxisMinorTickVisibilityOff()
            cube_axes.YAxisMinorTickVisibilityOff()
            cube_axes.ZAxisMinorTickVisibilityOff()
            # VTK's automatic ticks crowd stretched axes: label three points per
            # axis, or just the two ends of an axis much thinner than the others
            extents = [hi - lo for lo, hi in orig_ranges[:dim]]
            for axis, (lo, hi) in enumerate(orig_ranges[:dim]):
                thin = extents[axis] < 0.05 * max(extents)
                labels = vtk.vtkStringArray()
                for value in np.linspace(lo, hi, 2 if thin else 3):
                    labels.InsertNextValue(f"{value:.3g}")
                cube_axes.SetAxisLabels(axis, labels)

            if plot_type == "slice":
                if axis_idx == 0:
                    cube_axes.XAxisVisibilityOff()
                    cube_axes.SetXAxisTickVisibility(False)
                    cube_axes.SetXAxisLabelVisibility(False)
                elif axis_idx == 1:
                    cube_axes.YAxisVisibilityOff()
                    cube_axes.SetYAxisTickVisibility(False)
                    cube_axes.SetYAxisLabelVisibility(False)
                else:
                    cube_axes.ZAxisVisibilityOff()
                    cube_axes.SetZAxisTickVisibility(False)
                    cube_axes.SetZAxisLabelVisibility(False)

            ren.AddActor(cube_axes)

            window.AddRenderer(ren)
            all_renderers.append(ren)
            spatial_renderers.append(ren)

            # Camera setup: slice panels get independent orthographic cameras;
            # 3d panels share a perspective camera per set of grid bounds.
            if plot_type == "slice":
                ren.ResetCamera()
                cam = ren.GetActiveCamera()
                cam.SetParallelProjection(True)
                fp = list(cam.GetFocalPoint())
                gb = g.GetBounds()
                offset = (
                    max(
                        gb[1] - gb[0],
                        gb[3] - gb[2],
                        gb[5] - gb[4],
                    )
                    * 2
                )
                # Look from the negative side so OuterEdges places
                # axis labels on the top/left edges (more viewport room).
                # offset along the plane normal only, so the view is face-on
                pos = list(fp)
                pos[axis_idx] = fp[axis_idx] - offset
                cam.SetPosition(pos)
                view_up = [0, 0, 0]
                if axis_idx == 2:
                    view_up[1] = 1
                elif axis_idx == 1:
                    view_up[2] = 1
                else:
                    view_up[1] = 1
                cam.SetViewUp(view_up)
                ren.ResetCamera()
                cam.Zoom(0.70)
                cube_axes.SetCamera(cam)
            else:
                camera_key = tuple(np.round(g.GetBounds(), 12))
                cam = shared_cameras.get(camera_key)
                if cam is None:
                    ren.ResetCamera()
                    cam = ren.GetActiveCamera()
                    if dim == 3:
                        cam.Azimuth(-55)
                        cam.Elevation(25)
                    shared_cameras[camera_key] = cam
                else:
                    ren.SetActiveCamera(cam)
                cube_axes.SetCamera(cam)

            panel_idx += 1

        for name in self.scalar_names:
            vals = scalar_data[name]
            v_min, v_max = scalar_ranges[name]
            v_pad = max((v_max - v_min) * 0.05, 1e-10)

            chart = vtk.vtkChartXY()
            chart.SetTitle(name)
            chart.GetTitleProperties().SetFontSize(36)
            chart.GetTitleProperties().SetBold(True)
            chart.GetTitleProperties().SetColor(0, 0, 0)
            for axis_index, title in ((1, "Time [s]"), (0, name)):
                axis = chart.GetAxis(axis_index)
                axis.SetTitle(title)
                axis.GetTitleProperties().SetFontSize(28)
                axis.GetTitleProperties().SetColor(0, 0, 0)
                axis.GetLabelProperties().SetFontSize(22)
                axis.GetLabelProperties().SetColor(0, 0, 0)
            chart.GetAxis(1).SetRange(float(self.t_pts[0]), float(self.t_pts[-1]))
            chart.GetAxis(0).SetRange(v_min - v_pad, v_max + v_pad)

            table = vtk.vtkTable()
            t_arr = numpy_support.numpy_to_vtk(
                np.asarray(self.t_pts, dtype=np.float32), deep=True
            )
            t_arr.SetName("Time")
            v_arr = numpy_support.numpy_to_vtk(vals.astype(np.float32), deep=True)
            v_arr.SetName(name)
            table.AddColumn(t_arr)
            table.AddColumn(v_arr)

            line = chart.AddPlot(vtk.vtkChart.LINE)
            line.SetInputData(table, 0, 1)
            line.SetColor(31, 119, 180, 255)
            line.SetWidth(2.0)

            marker_table = vtk.vtkTable()
            mt_arr = vtk.vtkFloatArray()
            mt_arr.SetName("t")
            mv_arr = vtk.vtkFloatArray()
            mv_arr.SetName("v")
            mt_arr.InsertNextValue(float(self.t_pts[0]))
            mt_arr.InsertNextValue(float(self.t_pts[0]))
            mv_arr.InsertNextValue(v_min - v_pad)
            mv_arr.InsertNextValue(v_max + v_pad)
            marker_table.AddColumn(mt_arr)
            marker_table.AddColumn(mv_arr)

            marker_line = chart.AddPlot(vtk.vtkChart.LINE)
            marker_line.SetInputData(marker_table, 0, 1)
            marker_line.SetColor(200, 50, 50, 200)
            marker_line.SetWidth(1.5)
            time_markers.append((mt_arr, marker_table))

            view = vtk.vtkContextActor()
            scene = vtk.vtkContextScene()
            scene.AddItem(chart)
            view.SetScene(scene)

            ren = vtk.vtkRenderer()
            ren.AddActor(view)
            scene.SetRenderer(ren)
            ren.SetBackground(1, 1, 1)

            ren.SetViewport(*viewport(panel_idx))

            window.AddRenderer(ren)
            all_renderers.append(ren)
            chart_views.append((chart, view, scene))
            panel_idx += 1

        while panel_idx < n_rows * n_cols:
            ren = vtk.vtkRenderer()
            ren.SetBackground(1, 1, 1)
            ren.SetViewport(*viewport(panel_idx))
            window.AddRenderer(ren)
            panel_idx += 1

        slider_bg = vtk.vtkRenderer()
        slider_bg.SetBackground(1, 1, 1)
        slider_bg.SetViewport(0, 0, 1, slider_h)
        window.AddRenderer(slider_bg)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)

        time_text = vtk.vtkTextActor()
        time_text.SetInput(f"t = {self.t_pts[0]:.4g} s")
        time_text.GetTextProperty().SetFontSize(28)
        time_text.GetTextProperty().SetColor(0, 0, 0)
        time_text.GetTextProperty().SetBold(True)
        time_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        time_text.SetPosition(0.01, 0.15)
        slider_bg.AddViewProp(time_text)

        # Time slider — scaled in physical time (seconds)
        t_min = float(self.t_pts[0])
        t_max = float(self.t_pts[-1])
        slider_rep = vtk.vtkSliderRepresentation2D()
        slider_rep.SetMinimumValue(t_min)
        slider_rep.SetMaximumValue(t_max)
        slider_rep.SetValue(t_min)
        slider_rep.SetTitleText("")
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint1Coordinate().SetValue(0.15, slider_h * 0.5)
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint2Coordinate().SetValue(0.95, slider_h * 0.5)
        slider_rep.SetSliderLength(0.04)
        slider_rep.SetSliderWidth(0.06)
        slider_rep.SetTubeWidth(0.015)
        slider_rep.SetEndCapLength(0.02)
        slider_rep.SetEndCapWidth(0.06)
        slider_rep.GetTitleProperty().SetColor(0, 0, 0)
        slider_rep.GetLabelProperty().SetColor(0, 0, 0)
        slider_rep.GetLabelProperty().SetFontSize(16)
        slider_rep.GetSliderProperty().SetColor(0.2, 0.4, 0.8)
        slider_rep.GetTubeProperty().SetColor(0.7, 0.7, 0.7)
        slider_rep.GetCapProperty().SetColor(0.5, 0.5, 0.5)
        slider_rep.GetSelectedProperty().SetColor(0.3, 0.5, 0.9)

        _t_array = np.asarray(self.t_pts)

        def on_slider(obj, event):
            t_now = float(obj.GetRepresentation().GetValue())
            t_now = max(t_min, min(t_now, t_max))
            if not self.interpolate_time:
                t_idx = int(np.argmin(np.abs(_t_array - t_now)))
                # label the stored time that is drawn, not the raw slider value
                t_now = float(_t_array[t_idx])

            for sname, g, cell_to_point, cut in zip(
                panel_names,
                spatial_grids,
                cell_to_point_filters,
                cutters,
                strict=True,
            ):
                if self.interpolate_time:
                    vals = pv_by_name[sname](t_now).ravel()
                else:
                    vals = spatial_data[sname][:, t_idx]
                if is_cell_data_by_name[sname]:
                    _set_cell_scalars(g, sname, vals)
                else:
                    _set_point_scalars(g, sname, vals)
                if cell_to_point is not None:
                    cell_to_point.Modified()
                    cell_to_point.Update()
                if cut is not None:
                    cut.Update()

            for mt_arr, mtable in time_markers:
                mt_arr.SetValue(0, t_now)
                mt_arr.SetValue(1, t_now)
                mt_arr.Modified()
                mtable.Modified()
            time_text.SetInput(f"t = {t_now:.4g} s")
            if show_plot:  # pragma: no cover
                window.Render()

        slider = vtk.vtkSliderWidget()
        slider.SetInteractor(interactor)
        slider.SetRepresentation(slider_rep)
        slider.SetAnimationModeToAnimate()
        slider.EnabledOn()
        slider.AddObserver("InteractionEvent", on_slider)

        if show_plot:  # pragma: no cover
            interactor.Initialize()
            window.Render()
            interactor.Start()

        self._window = window
        self._interactor = interactor
        self._slider = slider
        self._time_text = time_text
        self._time_markers = [mt_arr for mt_arr, _ in time_markers]

    def save_gif(
        self,
        filename: str,
        fps: int = 10,
        n_frames: int = 100,
        width: int = 1800,
        height: int = 900,
    ) -> None:
        """Render an animation to a GIF file.

        Parameters
        ----------
        filename : str
            Output path (e.g. ``"anim.gif"``).
        fps : int
            Frames per second.
        n_frames : int
            Number of frames (evenly spaced in time).
        width, height : int
            Pixel dimensions of each frame.
        """
        vtk = pybamm.import_optional_dependency("vtk")
        Image = pybamm.import_optional_dependency("PIL.Image")

        if not hasattr(self, "_window") or not self._window.GetOffScreenRendering():
            self.dynamic_plot(show_plot=False)

        win = self._window
        win.SetOffScreenRendering(1)
        win.SetSize(width, height)

        t_min = float(self.t_pts[0])
        t_max = float(self.t_pts[-1])
        frame_times = np.linspace(t_min, t_max, n_frames)

        frames = []
        for t in frame_times:
            self._slider.GetRepresentation().SetValue(t)
            self._slider.InvokeEvent("InteractionEvent")
            win.Render()

            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(win)
            w2i.Update()
            img_data = w2i.GetOutput()

            w_px, h_px, _ = img_data.GetDimensions()
            n_comp = img_data.GetNumberOfScalarComponents()
            raw = np.frombuffer(
                memoryview(img_data.GetPointData().GetScalars()),
                dtype=np.uint8,
            ).reshape(h_px, w_px, n_comp)
            frames.append(Image.fromarray(raw[::-1]))

        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        pybamm.logger.info(f"Saved {len(frames)}-frame GIF to {filename}")
