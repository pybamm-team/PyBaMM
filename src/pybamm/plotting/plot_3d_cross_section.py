import numpy as np

import pybamm
from pybamm.util import import_optional_dependency


def plot_3d_cross_section(
    solution: "pybamm.Solution",
    variable: str,
    t: float | None,
    plane: str = "yz",
    position: float = 0.5,
    n_pts: int = 100,
    ax=None,
    show_plot: bool = True,
    cmap: str = "inferno",
    levels: int = 20,
    use_offset: bool = False,
    show_mesh: bool = False,
    mesh_color: str = "white",
    mesh_alpha: float = 0.4,
    mesh_linewidth: float = 0.7,
    mesh_tolerance: float | None = None,
    **kwargs,
):
    """
    Plots a high-quality 2D cross-section of a 3D variable from a PyBaMM solution,
    with mesh overlay support.

    Parameters
    ----------
    solution : pybamm.Solution
        The solution object containing the 3D variable.
    variable : str, optional
        The name of the 3D variable to plot (default: "Cell temperature [K]").
    t : float, optional
        The time at which to plot. If None, the last timestep is used.
    plane : str, optional
        The plane for the cross-section ('xy', 'yz', 'xz', 'rz').
    position : float, optional
        The relative position (0 to 1) along the third axis to take the slice.
    n_pts : int, optional
        The number of points for the plotting grid in each direction.
    ax : matplotlib.axes.Axes, optional
        The axes on which to draw the plot. If None, a new figure is created.
    show_plot : bool, optional
        Whether to display the plot.
    cmap : str, optional
        The colormap for the plot.
    levels : int, optional
        The number of contour levels for the plot.
    use_offset : bool, optional
        Parameter to control color bar format to use scientific notation (default: False).
    show_mesh : bool, optional
        Whether to overlay the calculated FEM mesh slice on the plot.
    mesh_color : str, optional
        Color of the mesh lines.
    mesh_alpha : float, optional
        Transparency of the mesh lines.
    mesh_linewidth : float, optional
        Width of the mesh lines.
    **kwargs
        Additional keyword arguments passed to matplotlib.contourf.
    """
    plt = import_optional_dependency("matplotlib.pyplot")
    model = solution.all_models[0]
    if model.options.get("dimensionality") != 3:
        raise TypeError("This function requires a 3D model solution.")

    if t is None:
        t = solution.t[-1]

    var_obj = solution[variable]
    mesh = var_obj.mesh
    nodes = mesh.nodes
    elements = mesh.elements

    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    z_min, z_max = np.min(nodes[:, 2]), np.max(nodes[:, 2])
    geometry = model.options.get("cell geometry", "pouch")

    if mesh_tolerance is None:
        domain_size = max(x_max - x_min, y_max - y_min, z_max - z_min)
        mesh_tolerance = domain_size * 0.01  # 1% of domain size

    fig = None
    if ax is None:
        if geometry == "cylindrical" and plane == "xy":
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

    fig = fig or ax.get_figure()
    title_suffix = f"at t={t:.1f}s"

    def add_mesh_overlay(
        ax, nodes, elements, plane, slice_coord_val, geometry, mesh_tolerance
    ):
        """Calculates and draws the true intersection of the mesh with the plane."""

        if plane == "yz":
            slice_axis_idx = 0  # slice on x
            plot_axes_indices = [1, 2]  # plot y,z
        elif plane == "xz":
            slice_axis_idx = 1  # slice on y
            plot_axes_indices = [0, 2]  # plot x,z
        elif plane == "xy":
            slice_axis_idx = 2  # slice on z
            plot_axes_indices = [0, 1]  # plot x,y
        elif plane == "rz":
            slice_axis_idx = 1  # slice on y
            slice_coord_val = 0.0

        mesh_segments = []

        for element in elements:
            intersection_points = []
            edges = [
                (element[0], element[1]),
                (element[0], element[2]),
                (element[0], element[3]),
                (element[1], element[2]),
                (element[1], element[3]),
                (element[2], element[3]),
            ]

            for p1_idx, p2_idx in edges:
                p1, p2 = nodes[p1_idx], nodes[p2_idx]
                c1, c2 = p1[slice_axis_idx], p2[slice_axis_idx]

                # Check if edge crosses the slice plane
                if abs(c1 - slice_coord_val) <= mesh_tolerance:
                    intersection_points.append(p1)
                elif abs(c2 - slice_coord_val) <= mesh_tolerance:
                    intersection_points.append(p2)
                elif (c1 < slice_coord_val < c2) or (c2 < slice_coord_val < c1):
                    # Edge crosses the plane
                    ratio = (slice_coord_val - c1) / (c2 - c1)
                    intersection_point = p1 + ratio * (p2 - p1)
                    intersection_points.append(intersection_point)

            # Remove duplicate points
            if len(intersection_points) >= 2:
                unique_points = []
                for pt in intersection_points:
                    is_duplicate = False
                    for existing_pt in unique_points:
                        if np.linalg.norm(pt - existing_pt) < mesh_tolerance:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_points.append(pt)

                if len(unique_points) >= 2:
                    poly_xyz = np.array(unique_points)

                    if geometry == "cylindrical" and plane == "xy":
                        x_coords, y_coords = poly_xyz[:, 0], poly_xyz[:, 1]
                        r_coords = np.sqrt(x_coords**2 + y_coords**2)
                        theta_coords = np.arctan2(y_coords, x_coords)
                        theta_coords[theta_coords < 0] += (
                            2 * np.pi
                        )  # Ensure positive angles
                        plot_coords = np.column_stack([theta_coords, r_coords])
                    elif geometry == "cylindrical" and plane == "rz":
                        mask = poly_xyz[:, 0] >= 0
                        if np.sum(mask) >= 2:
                            filtered_points = poly_xyz[mask]
                            r_coords = np.sqrt(
                                filtered_points[:, 0] ** 2 + filtered_points[:, 1] ** 2
                            )
                            z_coords = filtered_points[:, 2]
                            plot_coords = np.column_stack([r_coords, z_coords])
                        else:
                            continue
                    else:  # Cartesian
                        plot_coords = poly_xyz[:, plot_axes_indices]

                    if len(plot_coords) >= 2:
                        mesh_segments.append(plot_coords)

        segments_plotted = 0
        for segment in mesh_segments:
            if len(segment) >= 2:
                if len(segment) > 2:
                    # Sort points to form a proper polygon
                    centroid = np.mean(segment, axis=0)
                    angles = np.arctan2(
                        segment[:, 1] - centroid[1], segment[:, 0] - centroid[0]
                    )
                    sorted_segment = segment[np.argsort(angles)]
                    # Close the polygon
                    final_segment = np.vstack([sorted_segment, sorted_segment[0]])
                else:
                    final_segment = segment

                ax.plot(
                    final_segment[:, 0],
                    final_segment[:, 1],
                    color=mesh_color,
                    alpha=mesh_alpha,
                    linewidth=mesh_linewidth,
                )
                segments_plotted += 1

        print(f"Plotted {segments_plotted} mesh segments")

    x_label, y_label = "", ""

    slice_coord_val = None
    if geometry == "cylindrical":
        r_coords = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
        r_min, r_max = np.min(r_coords), np.max(r_coords)

        if plane == "xy":
            slice_coord_val = z_min + (z_max - z_min) * position
            r_grid = np.linspace(r_min, r_max, n_pts)
            theta_grid = np.linspace(0, 2 * np.pi, n_pts)
            R_mesh, Theta_mesh = np.meshgrid(r_grid, theta_grid)
            X_eval, Y_eval = R_mesh * np.cos(Theta_mesh), R_mesh * np.sin(Theta_mesh)
            Z_eval = np.full_like(X_eval, slice_coord_val)
            data = var_obj(t=t, x=X_eval, y=Y_eval, z=Z_eval)
            pcm = ax.contourf(
                Theta_mesh, R_mesh, data, levels=levels, cmap=cmap, **kwargs
            )
            ax.set_ylim(r_min, r_max)
            plot_title = f"T(r,θ) at z={slice_coord_val:.2f}m, {title_suffix}"

        elif plane == "rz":
            slice_coord_val = 0.0  # Slice at theta=0
            r_grid = np.linspace(r_min, r_max, n_pts)
            z_grid = np.linspace(z_min, z_max, n_pts)
            R_mesh, Z_mesh = np.meshgrid(r_grid, z_grid)
            X_eval, Y_eval = R_mesh, np.zeros_like(R_mesh)
            Z_eval = Z_mesh
            data = var_obj(t=t, x=X_eval, y=Y_eval, z=Z_eval)
            pcm = ax.contourf(R_mesh, Z_mesh, data, levels=levels, cmap=cmap, **kwargs)
            x_label, y_label = "Radius r [m]", "Height z [m]"
            plot_title = f"T(r,z) Cross-Section, {title_suffix}"
        else:
            raise ValueError(f"Plane '{plane}' invalid for cylindrical geometry.")

    else:  # Cartesian geometry
        if plane == "yz":
            slice_coord_val = x_min + (x_max - x_min) * position
            grid_1, grid_2 = (
                np.linspace(y_min, y_max, n_pts),
                np.linspace(z_min, z_max, n_pts),
            )
            Y_eval, Z_eval = np.meshgrid(grid_1, grid_2)
            X_eval = np.full_like(Y_eval, slice_coord_val)
            data = var_obj(
                t=t, x=X_eval.ravel(), y=Y_eval.ravel(), z=Z_eval.ravel()
            ).reshape(X_eval.shape)
            pcm = ax.contourf(Y_eval, Z_eval, data, levels=levels, cmap=cmap, **kwargs)
            x_label, y_label = "y [m]", "z [m]"
            plot_title = f"T(y,z) at x={slice_coord_val:.2f}m, {title_suffix}"
        elif plane == "xz":
            slice_coord_val = y_min + (y_max - y_min) * position
            grid_1, grid_2 = (
                np.linspace(x_min, x_max, n_pts),
                np.linspace(z_min, z_max, n_pts),
            )
            X_eval, Z_eval = np.meshgrid(grid_1, grid_2)
            Y_eval = np.full_like(X_eval, slice_coord_val)
            data = var_obj(
                t=t, x=X_eval.ravel(), y=Y_eval.ravel(), z=Z_eval.ravel()
            ).reshape(X_eval.shape)
            pcm = ax.contourf(X_eval, Z_eval, data, levels=levels, cmap=cmap, **kwargs)
            x_label, y_label = "x [m]", "z [m]"
            plot_title = f"T(x,z) at y={slice_coord_val:.2f}m, {title_suffix}"
        elif plane == "xy":
            slice_coord_val = z_min + (z_max - z_min) * position
            grid_1, grid_2 = (
                np.linspace(x_min, x_max, n_pts),
                np.linspace(y_min, y_max, n_pts),
            )
            X_eval, Y_eval = np.meshgrid(grid_1, grid_2)
            Z_eval = np.full_like(X_eval, slice_coord_val)
            data = var_obj(
                t=t, x=X_eval.ravel(), y=Y_eval.ravel(), z=Z_eval.ravel()
            ).reshape(X_eval.shape)
            pcm = ax.contourf(X_eval, Y_eval, data, levels=levels, cmap=cmap, **kwargs)
            x_label, y_label = "x [m]", "y [m]"
            plot_title = f"T(x,y) at z={slice_coord_val:.2f}m, {title_suffix}"
        else:
            raise ValueError(
                f"Plane '{plane}' invalid for Cartesian geometry. Use 'xy', 'yz', or 'xz'."
            )

    # Add mesh overlay
    if show_mesh:
        add_mesh_overlay(
            ax, nodes, elements, plane, slice_coord_val, geometry, mesh_tolerance
        )

    cbar = fig.colorbar(pcm, ax=ax, label=f"{variable}")
    if not use_offset:
        cbar.formatter.set_useOffset(False)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.set_aspect("auto", "box")

    if show_plot:
        plt.tight_layout()
        plt.show()

    return ax
