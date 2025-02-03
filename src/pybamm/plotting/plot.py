import pybamm
from .quick_plot import ax_min, ax_max
from pybamm.util import import_optional_dependency


def plot(x, y, ax=None, show_plot=True, **kwargs):
    """
    Generate a simple 1D plot. Calls `matplotlib.pyplot.plot` with keyword
    arguments 'kwargs'. For a list of 'kwargs' see the
    `matplotlib plot documentation
    <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_

    Parameters
    ----------
    x : :class:`pybamm.Array`
        The array to plot on the x axis
    y : :class:`pybamm.Array`
        The array to plot on the y axis
    ax : matplotlib Axis, optional
        The axis on which to put the plot. If None, a new figure and axis is created.
    show_plot : bool, optional
        Whether to show the plots. Default is True. Set to False if you want to
        only display the plot after plt.show() has been called.
    kwargs
        Keyword arguments, passed to plt.plot

    """
    plt = import_optional_dependency("matplotlib.pyplot")

    if not isinstance(x, pybamm.Array):
        raise TypeError("x must be 'pybamm.Array'")
    if not isinstance(y, pybamm.Array):
        raise TypeError("y must be 'pybamm.Array'")

    if ax is not None:
        show_plot = False
    else:
        _, ax = plt.subplots()

    ax.plot(x.entries, y.entries, **kwargs)
    ax.set_ylim([ax_min(y.entries), ax_max(y.entries)])

    if show_plot:  # pragma: no cover
        plt.show()

    return ax
