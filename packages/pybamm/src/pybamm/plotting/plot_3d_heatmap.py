import pybamm
from pybamm.util import import_optional_dependency


def plot_3d_heatmap(
    solution: "pybamm.Solution",
    variable: str,
    t: float | None,
    ax=None,
    show_plot: bool = True,
    cmap: str = "inferno",
    marker_size: float = 10,
    alpha: float = 0.7,
    use_offset: bool = False,
    **kwargs,
):
    """
    Creates a 3D scatter plot of a variable from a PyBaMM solution with a 3D mesh.

    This function visualizes the solution directly on the FEM nodes, providing a true
    representation of the 3D data.

    Parameters
    ----------
    solution : pybamm.Solution
        The solution object containing the 3D variable.
    variable : str, optional
        The name of the 3D variable to plot (default: "Cell temperature [K]").
    t : float, optional
        The time at which to plot. If None, the last timestep is used.
    ax : matplotlib.axes.Axes, optional
        A 3D axes object on which to draw the plot. If None, a new figure
        and axes are created.
    show_plot : bool, optional
        Whether to display the plot (default: True).
    cmap : str, optional
        The colormap for the plot (default: 'inferno').
    marker_size : float, optional
        The size of the markers in the scatter plot (default: 10).
    alpha : float, optional
        The transparency of the markers (0=transparent, 1=opaque). Default is 0.7.
    use_offset : bool, optional
        If True, uses scientific notation for the color bar (default: False).
    **kwargs
        Additional keyword arguments passed to matplotlib.axes.Axes.scatter.
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
    color_data = var_obj(t=t, x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2])

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")
    else:
        if ax.name != "3d":
            raise TypeError("The provided axes `ax` must be a 3D projection.")
        fig = ax.get_figure()

    scatter = ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        nodes[:, 2],
        c=color_data,
        cmap=cmap,
        s=marker_size,
        alpha=alpha,
        **kwargs,
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"3D Heatmap of {variable}\nat t={t:.1f}s")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, label=variable)
    if not use_offset:
        cbar.formatter.set_useOffset(False)
        fig.canvas.draw()

    ax.view_init(elev=20, azim=-65)
    fig.tight_layout()

    if show_plot:
        plt.show()

    return ax
