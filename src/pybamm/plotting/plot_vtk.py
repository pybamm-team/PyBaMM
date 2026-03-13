"""
VTK-based interactive visualization for unstructured mesh solutions.

Provides :class:`VTKQuickPlot`, a drop-in alternative to the matplotlib-based
:class:`QuickPlot` for 2D and 3D cell-centered FVM data on unstructured meshes.

Also supports 0D (time-series) variables rendered as VTK line charts.
"""

import numpy as np

import pybamm

_VTK_CELL_TYPE = {
    "triangle": 5,  # VTK_TRIANGLE
    "quad": 9,  # VTK_QUAD
    "tetrahedron": 10,  # VTK_TETRA
    "hexahedron": 12,  # VTK_HEXAHEDRON
}

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _build_vtk_grid(mesh, scale=None):
    """Build a ``vtkUnstructuredGrid`` from an ``UnstructuredSubMesh``."""
    import vtk

    nodes = mesh.nodes
    if scale is not None:
        nodes = nodes * np.asarray(scale)[: nodes.shape[1]]

    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(len(nodes))
    for i, nd in enumerate(nodes):
        if len(nd) == 2:
            pts.SetPoint(i, nd[0], nd[1], 0.0)
        else:
            pts.SetPoint(i, nd[0], nd[1], nd[2])

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(pts)

    cell_type = _VTK_CELL_TYPE[mesh.element_type]
    for cell in mesh.elements:
        id_list = vtk.vtkIdList()
        for v in cell:
            id_list.InsertNextId(int(v))
        grid.InsertNextCell(cell_type, id_list)

    return grid


def _compute_scale(mesh):
    """Per-axis scale factors that normalise coordinate spans to the largest."""
    nodes = mesh.nodes
    spans = np.array(
        [nodes[:, d].max() - nodes[:, d].min() for d in range(nodes.shape[1])]
    )
    max_span = spans.max()
    if max_span == 0:
        return np.ones(nodes.shape[1])
    return max_span / np.where(spans > 0, spans, max_span)


def _resolve_scale(scale_opt, mesh):
    """Turn a scale option into a concrete array or None."""
    if scale_opt == "auto":
        return _compute_scale(mesh)
    if scale_opt is None:
        return None
    return np.asarray(scale_opt)


def _set_cell_scalars(grid, name, values):
    """Set (or update) a cell scalar array on a VTK grid."""
    import vtk

    arr = grid.GetCellData().GetArray(name)
    if arr is None:
        arr = vtk.vtkFloatArray()
        arr.SetName(name)
        arr.SetNumberOfTuples(len(values))
        grid.GetCellData().AddArray(arr)
        grid.GetCellData().SetActiveScalars(name)
    for i, v in enumerate(values):
        arr.SetValue(i, float(v))
    arr.Modified()
    grid.Modified()


def _viridis_lut(vmin, vmax, n=256):
    """Build a VTK lookup table using the matplotlib viridis colormap."""
    import vtk

    try:
        from matplotlib.cm import viridis as _cmap
    except ImportError:
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


class VTKQuickPlot:
    """Interactive VTK visualization for unstructured FVM solutions.

    Supports spatial (unstructured 2D/3D) and 0D (time-series) variables.

    Parameters
    ----------
    solutions : :class:`pybamm.Solution` or list thereof
    output_variables : list of str
    options : dict, optional
        Per-variable options keyed by variable name.  Each value is a dict
        that may contain:

        - ``"plot_type"``: ``"3d"`` (default) or ``"slice"``
        - ``"x"`` / ``"y"`` / ``"z"``: float in [0, 1] giving the slice
          position as a fraction of the axis range (required when
          ``plot_type`` is ``"slice"``)
        - ``"scale"``: ``"auto"`` (default), ``None``, or ``(sx, sy, sz)``

    Each variable's options value may also be a **list** of dicts, in which
    case one panel is created per entry::

        options={"T": [
            {"plot_type": "3d"},
            {"plot_type": "slice", "x": 0.5},
        ]}
    """

    def __init__(
        self,
        solutions,
        output_variables=None,
        options=None,
        interpolate_time=False,
    ):
        if isinstance(solutions, pybamm.Simulation):
            solutions = solutions.solution
        if isinstance(solutions, pybamm.Solution):
            solutions = [solutions]
        self.solution = solutions[0]

        if output_variables is None:
            output_variables = list(self.solution.all_models[0].variables.keys())[:1]
        if isinstance(output_variables, str):
            output_variables = [output_variables]

        self.spatial_names = []
        self.spatial_vars = []
        self.scalar_names = []
        self.scalar_vars = []

        for name in output_variables:
            pv = self.solution[name]
            if isinstance(pv, pybamm.ProcessedVariableUnstructuredFVM):
                self.spatial_names.append(name)
                self.spatial_vars.append(pv)
            else:
                self.scalar_names.append(name)
                self.scalar_vars.append(pv)

        self.output_variables = output_variables
        self.mesh = self.spatial_vars[0].mesh if self.spatial_vars else None
        self.t_pts = self.solution.t
        self.interpolate_time = interpolate_time

        _defaults = {"plot_type": "3d", "scale": "auto"}
        raw_opts = options or {}

        # Build spatial_panels: flat list of (name, opts_dict) tuples.
        self.spatial_panels = []
        for name in self.spatial_names:
            var_opt = raw_opts.get(name, _defaults)
            if isinstance(var_opt, dict):
                opt_list = [var_opt]
            else:
                opt_list = list(var_opt)
            for single_opt in opt_list:
                merged = dict(_defaults)
                merged.update(single_opt)
                self.spatial_panels.append((name, merged))

    # ------------------------------------------------------------------

    def dynamic_plot(self, show_plot=True):
        """Launch an interactive VTK window with a time slider."""
        import vtk

        n_spatial = len(self.spatial_panels)
        n_scalar = len(self.scalar_names)
        n_panels = n_spatial + n_scalar

        # --- Precompute spatial data ---
        spatial_data = {}
        spatial_mins = {}
        spatial_maxs = {}
        for name, pv in zip(self.spatial_names, self.spatial_vars, strict=True):
            pv.initialise()
            data = np.column_stack([pv._data_at_time(t).ravel() for t in self.t_pts])
            spatial_data[name] = data
            spatial_mins[name] = float(data.min())
            spatial_maxs[name] = float(data.max())

        # --- Precompute scalar (0D) data ---
        scalar_data = {}
        for name, pv in zip(self.scalar_names, self.scalar_vars, strict=True):
            pv.initialise()
            if isinstance(pv, pybamm.ProcessedVariableUnstructuredFVM):
                vals = np.array(
                    [float(pv._data_at_time(t).ravel()[0]) for t in self.t_pts]
                )
            else:
                vals = np.array([float(pv(t).ravel()[0]) for t in self.t_pts])
            scalar_data[name] = vals

        # --- Layout ---
        slider_h = 0.08
        panel_top = 1.0
        panel_bot = slider_h

        n_cols = int(np.ceil(np.sqrt(n_panels)))
        n_rows = int(np.ceil(n_panels / n_cols))
        panel_height = (panel_top - panel_bot) / n_rows

        window = vtk.vtkRenderWindow()
        window.SetSize(650 * n_cols, 520 * n_rows)
        window.SetWindowName("PyBaMM - " + ", ".join(self.output_variables))

        all_renderers = []
        spatial_grids = []
        c2p_filters = []
        cutters = []
        chart_views = []
        time_markers = []

        panel_idx = 0

        # --- Spatial panels ---
        first_3d_cam = None
        spatial_renderers = []
        panel_names = []

        for name, opts in self.spatial_panels:
            plot_type = opts.get("plot_type", "3d")
            var_scale = _resolve_scale(opts.get("scale", "auto"), self.mesh)
            panel_names.append(name)

            g = _build_vtk_grid(self.mesh, scale=var_scale)
            _set_cell_scalars(g, name, spatial_data[name][:, 0])
            spatial_grids.append(g)

            c2p = vtk.vtkCellDataToPointData()
            c2p.SetInputData(g)
            c2p.Update()
            c2p_filters.append(c2p)

            # Determine pipeline source: cutter for slices, c2p for 3d
            cutter = None
            if plot_type == "slice":
                axis_key = None
                for ak in ("x", "y", "z"):
                    if ak in opts:
                        axis_key = ak
                        break
                if axis_key is None:
                    raise ValueError(
                        f"plot_type='slice' for '{name}' requires one of "
                        f"'x', 'y', or 'z' specifying the slice fraction"
                    )
                axis_idx = _AXIS_INDEX[axis_key]
                frac = float(opts[axis_key])
                nodes = self.mesh.nodes
                lo = float(nodes[:, axis_idx].min())
                hi = float(nodes[:, axis_idx].max())
                phys_val = lo + frac * (hi - lo)
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
                cutter.SetInputConnection(c2p.GetOutputPort())
                cutter.Update()

                mapper_source = cutter.GetOutputPort()
            else:
                mapper_source = c2p.GetOutputPort()

            cutters.append(cutter)

            lut = _viridis_lut(spatial_mins[name], spatial_maxs[name])

            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputConnection(mapper_source)
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
            sb.SetWidth(0.18)
            sb.SetHeight(0.5)
            sb.SetPosition(0.80, 0.25)
            sb.GetLabelTextProperty().SetFontSize(24)
            sb.GetLabelTextProperty().SetColor(0, 0, 0)
            sb.SetUnconstrainedFontSize(True)
            sb.SetLabelFormat("%-#6.3g")

            title_actor = vtk.vtkTextActor()
            title_actor.SetInput(name)
            title_actor.GetTextProperty().SetFontSize(36)
            title_actor.GetTextProperty().SetColor(0, 0, 0)
            title_actor.GetTextProperty().SetBold(True)
            title_actor.GetTextProperty().SetJustificationToCentered()
            title_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            title_actor.SetPosition(0.5, 0.92)

            ren = vtk.vtkRenderer()
            ren.AddActor(actor)
            ren.AddActor2D(sb)
            ren.AddActor2D(title_actor)
            ren.SetBackground(1, 1, 1)

            row = panel_idx // n_cols
            col = panel_idx % n_cols
            y0 = panel_top - (row + 1) * panel_height
            y1 = panel_top - row * panel_height
            ren.SetViewport(col / n_cols, y0, (col + 1) / n_cols, y1)

            # Cube axes
            if self.mesh is not None:
                mesh_nodes = self.mesh.nodes
                dim = mesh_nodes.shape[1]

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
                cube_axes.SetScreenSize(10.0)
                cube_axes.SetLabelOffset(10)
                cube_axes.SetTitleOffset([20, 20])

                orig_ranges = [
                    (float(mesh_nodes[:, d].min()), float(mesh_nodes[:, d].max()))
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
                    tp.SetFontSize(28)
                    tp.SetColor(0.15, 0.15, 0.15)
                    tp.SetBold(True)
                    lp = cube_axes.GetLabelTextProperty(ax_id)
                    lp.SetFontSize(22)
                    lp.SetColor(0.25, 0.25, 0.25)
                cube_axes.SetXTitle("X")
                cube_axes.SetYTitle("Y")
                cube_axes.SetZTitle("Z")
                cube_axes.SetXLabelFormat("%.2g")
                cube_axes.SetYLabelFormat("%.2g")
                cube_axes.SetZLabelFormat("%.2g")
                cube_axes.XAxisMinorTickVisibilityOff()
                cube_axes.YAxisMinorTickVisibilityOff()
                cube_axes.ZAxisMinorTickVisibilityOff()

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
            # 3d panels share a single perspective camera.
            if plot_type == "slice":
                ren.ResetCamera()
                cam = ren.GetActiveCamera()
                cam.SetParallelProjection(True)
                pos = list(cam.GetPosition())
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
                if self.mesh is not None:
                    cube_axes.SetCamera(cam)
            else:
                if first_3d_cam is None:
                    ren.ResetCamera()
                    first_3d_cam = ren.GetActiveCamera()
                    if self.mesh is not None and self.mesh.dimension == 3:
                        first_3d_cam.Azimuth(-55)
                        first_3d_cam.Elevation(25)
                    if self.mesh is not None:
                        cube_axes.SetCamera(first_3d_cam)
                else:
                    ren.SetActiveCamera(first_3d_cam)
                    if self.mesh is not None:
                        cube_axes.SetCamera(first_3d_cam)

            panel_idx += 1

        # --- Scalar (0D chart) panels ---
        for name in self.scalar_names:
            vals = scalar_data[name]
            v_min, v_max = float(vals.min()), float(vals.max())
            v_pad = max((v_max - v_min) * 0.05, 1e-10)

            chart = vtk.vtkChartXY()
            chart.SetTitle(name)
            chart.GetTitleProperties().SetFontSize(36)
            chart.GetTitleProperties().SetBold(True)
            chart.GetTitleProperties().SetColor(0, 0, 0)
            chart.GetAxis(1).SetTitle("Time [s]")
            chart.GetAxis(0).SetTitle(name)
            chart.GetAxis(1).GetTitleProperties().SetFontSize(28)
            chart.GetAxis(1).GetTitleProperties().SetColor(0, 0, 0)
            chart.GetAxis(1).GetLabelProperties().SetFontSize(22)
            chart.GetAxis(1).GetLabelProperties().SetColor(0, 0, 0)
            chart.GetAxis(0).GetTitleProperties().SetFontSize(28)
            chart.GetAxis(0).GetTitleProperties().SetColor(0, 0, 0)
            chart.GetAxis(0).GetLabelProperties().SetFontSize(22)
            chart.GetAxis(0).GetLabelProperties().SetColor(0, 0, 0)
            chart.GetAxis(1).SetRange(float(self.t_pts[0]), float(self.t_pts[-1]))
            chart.GetAxis(0).SetRange(v_min - v_pad, v_max + v_pad)

            table = vtk.vtkTable()
            t_arr = vtk.vtkFloatArray()
            t_arr.SetName("Time")
            v_arr = vtk.vtkFloatArray()
            v_arr.SetName(name)
            for i in range(len(self.t_pts)):
                t_arr.InsertNextValue(float(self.t_pts[i]))
                v_arr.InsertNextValue(float(vals[i]))
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

            row = panel_idx // n_cols
            col = panel_idx % n_cols
            y0 = panel_top - (row + 1) * panel_height
            y1 = panel_top - row * panel_height
            ren.SetViewport(col / n_cols, y0, (col + 1) / n_cols, y1)

            window.AddRenderer(ren)
            all_renderers.append(ren)
            chart_views.append((chart, view, scene))
            panel_idx += 1

        # --- Fill any unused grid cells with white ---
        while panel_idx < n_rows * n_cols:
            ren = vtk.vtkRenderer()
            ren.SetBackground(1, 1, 1)
            row = panel_idx // n_cols
            col = panel_idx % n_cols
            y0 = panel_top - (row + 1) * panel_height
            y1 = panel_top - row * panel_height
            ren.SetViewport(col / n_cols, y0, (col + 1) / n_cols, y1)
            window.AddRenderer(ren)
            panel_idx += 1

        # --- Slider background (white strip at bottom) ---
        slider_bg = vtk.vtkRenderer()
        slider_bg.SetBackground(1, 1, 1)
        slider_bg.SetViewport(0, 0, 1, slider_h)
        window.AddRenderer(slider_bg)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)

        # Time label
        time_text = vtk.vtkTextActor()
        time_text.SetInput(f"t = {self.t_pts[0]:.4g} s")
        time_text.GetTextProperty().SetFontSize(28)
        time_text.GetTextProperty().SetColor(0, 0, 0)
        time_text.GetTextProperty().SetBold(True)
        time_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        time_text.SetPosition(0.01, 0.15)
        slider_bg.AddActor2D(time_text)

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

        # Look-up table for snapping to nearest timestep
        _t_array = np.asarray(self.t_pts)

        # Keep references for interpolated mode
        _spatial_vars = {
            name: pv
            for name, pv in zip(
                self.spatial_names,
                self.spatial_vars,
                strict=True,
            )
        }

        def on_slider(obj, event):
            t_now = float(obj.GetRepresentation().GetValue())
            t_now = max(t_min, min(t_now, t_max))

            if self.interpolate_time:
                # Evaluate every spatial variable at exact time
                for sname, g, c2p, cut in zip(
                    panel_names,
                    spatial_grids,
                    c2p_filters,
                    cutters,
                    strict=True,
                ):
                    vals = _spatial_vars[sname]._data_at_time(t_now).ravel()
                    _set_cell_scalars(g, sname, vals)
                    c2p.Modified()
                    c2p.Update()
                    if cut is not None:
                        cut.Update()
            else:
                # Snap to nearest stored timestep (fast)
                t_idx = int(np.argmin(np.abs(_t_array - t_now)))
                for sname, g, c2p, cut in zip(
                    panel_names,
                    spatial_grids,
                    c2p_filters,
                    cutters,
                    strict=True,
                ):
                    _set_cell_scalars(g, sname, spatial_data[sname][:, t_idx])
                    c2p.Modified()
                    c2p.Update()
                    if cut is not None:
                        cut.Update()

            for mt_arr, mtable in time_markers:
                mt_arr.SetValue(0, t_now)
                mt_arr.SetValue(1, t_now)
                mt_arr.Modified()
                mtable.Modified()
            time_text.SetInput(f"t = {t_now:.4g} s")
            window.Render()

        slider = vtk.vtkSliderWidget()
        slider.SetInteractor(interactor)
        slider.SetRepresentation(slider_rep)
        slider.SetAnimationModeToAnimate()
        slider.EnabledOn()
        slider.AddObserver("InteractionEvent", on_slider)

        if show_plot:
            interactor.Initialize()
            window.Render()
            interactor.Start()

        self._window = window
        self._interactor = interactor
        self._slider = slider

    def save_gif(self, filename, fps=10, n_frames=100, width=1800, height=900):
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
        import vtk
        from PIL import Image

        if not hasattr(self, "_window"):
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
        print(f"Saved {len(frames)}-frame GIF to {filename}")
