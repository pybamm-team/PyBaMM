import pybamm
from .quick_plot import ax_min, ax_max
from pybamm.util import import_optional_dependency


def plot2D(x, y, z, ax=None, show_plot=True, **kwargs):
    """
    Generate a simple 2D plot. Calls `matplotlib.pyplot.contourf` with keyword
    arguments 'kwargs'.  For a list of 'kwargs' see the
    `matplotlib contourf documentation
    <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html>`_

    Parameters
    ----------
    x : :class:`pybamm.Array`
        The array to plot on the x axis. Can be of shape (M, N) or (N, 1)
    y : :class:`pybamm.Array`
        The array to plot on the y axis. Can be of shape (M, N)  or (M, 1)
    z : :class:`pybamm.Array`
        The array to plot on the z axis. Is of shape (M, N)
    ax : matplotlib Axis, optional
        The axis on which to put the plot. If None, a new figure and axis is created.
    show_plot : bool, optional
        Whether to show the plots. Default is True. Set to False if you want to
        only display the plot after plt.show() has been called.

    """
    plt = import_optional_dependency("matplotlib.pyplot")

    if not isinstance(x, pybamm.Array):
        raise TypeError("x must be 'pybamm.Array'")
    if not isinstance(y, pybamm.Array):
        raise TypeError("y must be 'pybamm.Array'")
    if not isinstance(z, pybamm.Array):
        raise TypeError("z must be 'pybamm.Array'")

    if ax is not None:
        show_plot = False
    else:
        _, ax = plt.subplots()

    # Get correct entries of x and y depending on shape
    if x.shape == y.shape == z.shape:
        x_entries = x.entries
        y_entries = y.entries
    else:
        x_entries = x.entries[:, 0]
        y_entries = y.entries[:, 0]

    plot = ax.contourf(
        x_entries,
        y_entries,
        z.entries,
        vmin=ax_min(z.entries),
        vmax=ax_max(z.entries),
        **kwargs,
    )
    plt.colorbar(plot, ax=ax)

    if show_plot:  # pragma: no cover
        plt.show()

    return ax
