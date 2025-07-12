import matplotlib.pyplot as plt
import numpy as np

import pybamm


def plot_3d_cross_section(
    solution: "pybamm.Solution",
    variable: str = "Cell temperature [K]",
    t: float | None = None,
    plane: str = "xy",
    position: float = 0.5,
    n_pts: int = 100,
    ax=None,
    show_plot: bool = True,
    cmap: str = "inferno",
    levels: int = 20,
    use_offset: bool = False,
    **kwargs,
):
    """
    Plots a high-quality 2D cross-section of a 3D variable from a PyBaMM solution.

    This function generates clear contour plots for both Cartesian and cylindrical
    geometries, automatically creating polar plots for xy-slices of cylinders.

    Parameters
    ----------
    solution : pybamm.Solution
        The solution object containing the 3D variable.
    variable : str, optional
        The name of the 3D variable to plot (default: "Cell temperature [K]").
    t : float, optional
        The time at which to plot. If None, the last timestep is used.
    plane : str, optional
        The plane for the cross-section ('xy', 'yz', 'xz'). For cylindrical
        geometries, 'rz' is also available (default: 'xy').
    position : float, optional
        The relative position (0 to 1) along the third axis to take the slice
        (default: 0.5).
    n_pts : int, optional
        The number of points for the plotting grid in each direction (default: 100).
    ax : matplotlib.axes.Axes, optional
        The axes on which to draw the plot. If None, a new figure is created.
    show_plot : bool, optional
        Whether to display the plot (default: True).
    cmap : str, optional
        The colormap for the plot (default: 'inferno').
    levels : int, optional
        The number of contour levels for the plot (default: 20).
    use_offset : bool, optional
        Parameter to control color bar format to use scientific notation (default: False).
    **kwargs
        Additional keyword arguments passed to matplotlib.contourf.
    """

    model = solution.all_models[0]
    if model.options.get("dimensionality") != 3:
        raise TypeError("This function requires a 3D model solution.")

    if t is None:
        t = solution.t[-1]

    var_obj = solution[variable]
    mesh = var_obj.mesh
    nodes = mesh.nodes
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    z_min, z_max = np.min(nodes[:, 2]), np.max(nodes[:, 2])
    geometry = model.options.get("cell geometry", "cartesian")

    fig = None
    if ax is None:
        if geometry == "cylindrical" and plane == "xy":
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

    fig = fig or ax.get_figure()
    title_suffix = f"at t={t:.1f}s"

    if geometry == "cylindrical":
        r_coords = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
        r_min, r_max = np.min(r_coords), np.max(r_coords)

        if plane == "xy":
            slice_coord_val = z_min + (z_max - z_min) * position
            r_grid = np.linspace(r_min, r_max, n_pts)
            theta_grid = np.linspace(0, 2 * np.pi, n_pts)
            R_mesh, Theta_mesh = np.meshgrid(r_grid, theta_grid)
            X_eval = R_mesh * np.cos(Theta_mesh)
            Y_eval = R_mesh * np.sin(Theta_mesh)
            Z_eval = np.full_like(X_eval, slice_coord_val)

            data = var_obj(t=t, x=X_eval, y=Y_eval, z=Z_eval)

            pcm = ax.contourf(
                Theta_mesh, R_mesh, data, levels=levels, cmap=cmap, **kwargs
            )
            ax.set_ylim(r_min, r_max)
            ax.set_title(f"T(r,θ) at z={slice_coord_val:.2f}m, {title_suffix}")

            cbar = fig.colorbar(pcm, ax=ax, label=variable)
            if not use_offset:
                cbar.formatter.set_useOffset(False)

            if show_plot:
                plt.tight_layout()
                plt.show()
            return ax

        elif plane == "rz":
            r_grid = np.linspace(r_min, r_max, n_pts)
            z_grid = np.linspace(z_min, z_max, n_pts)
            R_mesh, Z_mesh = np.meshgrid(r_grid, z_grid)
            X_eval, Y_eval = R_mesh, np.zeros_like(R_mesh)
            Z_eval = Z_mesh
            x_label, y_label = "Radius r [m]", "Height z [m]"
            plot_title = f"T(r,z) Cross-Section, {title_suffix}"
            data_X, data_Y = R_mesh, Z_mesh
    else:
        if plane == "yz":
            slice_coord_val = x_min + (x_max - x_min) * position
            y_grid, z_grid = (
                np.linspace(y_min, y_max, n_pts),
                np.linspace(z_min, z_max, n_pts),
            )
            Y_eval, Z_eval = np.meshgrid(y_grid, z_grid)
            X_eval = np.full_like(Y_eval, slice_coord_val)
            x_label, y_label = "y [m]", "z [m]"
            plot_title = f"T(y,z) at x={slice_coord_val:.2f}m, {title_suffix}"
            data_X, data_Y = Y_eval, Z_eval
        elif plane == "xz":
            slice_coord_val = y_min + (y_max - y_min) * position
            x_grid, z_grid = (
                np.linspace(x_min, x_max, n_pts),
                np.linspace(z_min, z_max, n_pts),
            )
            X_eval, Z_eval = np.meshgrid(x_grid, z_grid)
            Y_eval = np.full_like(X_eval, slice_coord_val)
            x_label, y_label = "x [m]", "z [m]"
            plot_title = f"T(x,z) at y={slice_coord_val:.2f}m, {title_suffix}"
            data_X, data_Y = X_eval, Z_eval
        elif plane == "xy":
            slice_coord_val = z_min + (z_max - z_min) * position
            x_grid, y_grid = (
                np.linspace(x_min, x_max, n_pts),
                np.linspace(y_min, y_max, n_pts),
            )
            X_eval, Y_eval = np.meshgrid(x_grid, y_grid)
            Z_eval = np.full_like(X_eval, slice_coord_val)
            x_label, y_label = "x [m]", "y [m]"
            plot_title = f"T(x,y) at z={slice_coord_val:.2f}m, {title_suffix}"
            data_X, data_Y = X_eval, Y_eval
        elif plane not in ["rz"]:
            raise ValueError(f"Plane '{plane}' invalid. Use 'xy', 'yz', 'xz', or 'rz'.")

    data_cross_section = var_obj(
        t=t, x=X_eval.ravel(), y=Y_eval.ravel(), z=Z_eval.ravel()
    ).reshape(X_eval.shape)

    pcm = ax.contourf(
        data_X, data_Y, data_cross_section, levels=levels, cmap=cmap, **kwargs
    )

    cbar = fig.colorbar(pcm, ax=ax, label=variable)
    if not use_offset:
        cbar.formatter.set_useOffset(False)
        # Forcing a redraw ensures the formatter update is applied before showing
        fig.canvas.draw()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.set_aspect("auto", "box")

    if show_plot:
        plt.tight_layout()
        plt.show()

    return ax
