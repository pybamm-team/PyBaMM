#
# Method for creating a filled contour plot of pybamm arrays
#
import pybamm
from .quick_plot import ax_min, ax_max


def plot2D(x, y, z, xlabel=None, ylabel=None, title=None, testing=False, **kwargs):
    """
    Generate a simple 2D plot. Calls `matplotlib.pyplot.contourf` with keyword
    arguments 'kwargs'.  For a list of 'kwargs' see the
    `matplotlib contourf documentation <https://tinyurl.com/y8mnadtn>`_

    Parameters
    ----------
    x : :class:`pybamm.Array`
        The array to plot on the x axis. Can be of shape (M, N) or (N, 1)
    y : :class:`pybamm.Array`
        The array to plot on the y axis. Can be of shape (M, N)  or (M, 1)
    z : :class:`pybamm.Array`
        The array to plot on the z axis. Is of shape (M, N)
    xlabel : str, optional
        The label for the x axis
    ylabel : str, optional
        The label for the y axis
    title : str, optional
        The title for the plot
    testing : bool, optional
        Whether to actually make the plot (turned off for unit tests)

    """
    import matplotlib.pyplot as plt

    if not isinstance(x, pybamm.Array):
        raise TypeError("x must be 'pybamm.Array'")
    if not isinstance(y, pybamm.Array):
        raise TypeError("y must be 'pybamm.Array'")
    if not isinstance(z, pybamm.Array):
        raise TypeError("z must be 'pybamm.Array'")

    # Get correct entries of x and y depending on shape
    if x.shape == y.shape == z.shape:
        x_entries = x.entries
        y_entries = y.entries
    else:
        x_entries = x.entries[:, 0]
        y_entries = y.entries[:, 0]

    plt.contourf(
        x_entries,
        y_entries,
        z.entries,
        vmin=ax_min(z.entries),
        vmax=ax_max(z.entries),
        **kwargs
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()

    if not testing:  # pragma: no cover
        plt.show()

    return
