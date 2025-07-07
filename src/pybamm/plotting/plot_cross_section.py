import numpy as np
import matplotlib.pyplot as plt

def plot_cross_section(
    solution,
    variable="Cell temperature [K]",
    t=None,
    plane="xy",
    position=0.5,
    n_pts=100,
    ax=None,
    show_plot=True,
    **kwargs,
):
    """
    Plots a high-quality 2D cross-section of a 3D variable from a PyBaMM solution.

    This function uses imshow for a high-quality heatmap and is robust to
    different parameter sets and geometries by deriving dimensions from the mesh.
    It extracts data directly from the solution vector to ensure correctness.

    Parameters
    ----------
    solution : pybamm.Solution
        The solution object containing the variable.
    variable : str, optional
        The name of the 3D variable to plot.
    t : float, optional
        The time at which to plot. If None, the last timestep is used.
    plane : str, optional
        The plane for the cross-section. Options are 'xy', 'yz', 'xz'.
        For cylindrical geometries, a special 'rz' plane is also available.
    position : float, optional
        The relative position (from 0 to 1) along the third axis to take the slice.
    n_pts : int, optional
        The number of points to use in each direction for the plotting grid.
    ax : matplotlib.axes.Axes, optional
        The axes on which to draw the plot. If None, a new figure and axes are created.
    show_plot : bool, optional
        Whether to show the plot.
    **kwargs
        Additional keyword arguments passed to matplotlib.imshow.
    """
    if t is None:
        t = solution.t[-1]

    model = solution.all_models[0]
    geometry = model.options["cell geometry"]
    var_3d = solution[variable]
    mesh = var_3d.mesh
    nodes = mesh.nodes

    from scipy.interpolate import interp1d, LinearNDInterpolator

    var_slice = model.variables[variable].y_slices[0]
    raw_data = solution.y[var_slice]

    time_interpolator = interp1d(
        solution.t, raw_data, kind="linear", axis=1, bounds_error=False, fill_value="extrapolate"
    )
    data_at_t = time_interpolator(t)

    spatial_interpolator = LinearNDInterpolator(points=nodes, values=data_at_t)

    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    z_min, z_max = np.min(nodes[:, 2]), np.max(nodes[:, 2])
    
    if geometry == "cylindrical" and plane == "rz":
        r_coords = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
        r_min, r_max = np.min(r_coords), np.max(r_coords)
        r_grid = np.linspace(r_min, r_max, n_pts)
        z_grid = np.linspace(z_min, z_max, n_pts)
        R_mesh, Z_mesh = np.meshgrid(r_grid, z_grid)
        X_eval, Y_eval, Z_eval = R_mesh, np.zeros_like(R_mesh), Z_mesh
        plot_extent = [r_min * 1000, r_max * 1000, z_min * 1000, z_max * 1000]
        x_label, y_label = "Radius (r) [mm]", "Height (z) [mm]"
        data_to_plot_shape = R_mesh.T # Transpose for correct imshow orientation
    else:
        if plane == "yz":
            slice_coord = x_min + (x_max - x_min) * position
            y_grid, z_grid = np.linspace(y_min, y_max, n_pts), np.linspace(z_min, z_max, n_pts)
            Y_eval, Z_eval = np.meshgrid(y_grid, z_grid)
            X_eval = np.full_like(Y_eval, slice_coord)
            plot_extent = [y_min * 1000, y_max * 1000, z_min * 1000, z_max * 1000]
            x_label, y_label = "y-axis [mm]", "z-axis [mm]"
            data_to_plot_shape = Y_eval.T # Transpose for correct imshow orientation
        elif plane == "xz":
            slice_coord = y_min + (y_max - y_min) * position
            x_grid, z_grid = np.linspace(x_min, x_max, n_pts), np.linspace(z_min, z_max, n_pts)
            X_eval, Z_eval = np.meshgrid(x_grid, z_grid)
            Y_eval = np.full_like(X_eval, slice_coord)
            plot_extent = [x_min * 1000, x_max * 1000, z_min * 1000, z_max * 1000]
            x_label, y_label = "x-axis [mm]", "z-axis [mm]"
            data_to_plot_shape = X_eval.T # Transpose for correct imshow orientation
        elif plane == "xy":
            slice_coord = z_min + (z_max - z_min) * position
            x_grid, y_grid = np.linspace(x_min, x_max, n_pts), np.linspace(y_min, y_max, n_pts)
            X_eval, Y_eval = np.meshgrid(x_grid, y_grid)
            Z_eval = np.full_like(X_eval, slice_coord)
            plot_extent = [x_min * 1000, x_max * 1000, y_min * 1000, y_max * 1000]
            x_label, y_label = "x-axis [mm]", "y-axis [mm]"
            data_to_plot_shape = X_eval.T # Transpose for correct imshow orientation
        else:
            raise ValueError(f"Plane '{plane}' must be one of 'xy', 'yz', 'xz', or 'rz' for cylinders.")

    eval_points = np.column_stack([X_eval.ravel(), Y_eval.ravel(), Z_eval.ravel()])
    data_cross_section = spatial_interpolator(eval_points).reshape(data_to_plot_shape.shape)

    if geometry == "cylindrical" and plane == "xy":
        r_outer = np.max(np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2))
        r_inner = np.min(np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2))
        x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2
        radius_sq = (X_eval - x_center)**2 + (Y_eval - y_center)**2
        mask = (radius_sq > r_outer**2) | (radius_sq < r_inner**2)
        data_cross_section[mask.T] = np.nan # Transpose mask to match data

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6.5))
    else:
        fig = ax.get_figure()

    kwargs.pop('levels', None)
    if 'cmap' not in kwargs:
        kwargs['cmap'] = "inferno"
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'bilinear'
    
    im = ax.imshow(data_cross_section, extent=plot_extent, origin='lower', **kwargs)
    fig.colorbar(im, ax=ax, label=f"{variable}")
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Cross-section at t={t:.0f}s")
    
    ax.set_aspect('equal', 'box')

    if show_plot:
        plt.tight_layout()
        plt.show()

    return ax
