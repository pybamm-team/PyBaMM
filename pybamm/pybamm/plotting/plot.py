#
# Method for creating a 1D plot of pybamm arrays
#
import pybamm
from .quick_plot import ax_min, ax_max
from pybamm.util import have_optional_dependency


def plot(x, y, ax=None, testing=False, **kwargs):
    """
    Generate a simple 1D plot. Calls `matplotlib.pyplot.plot` with keyword
    arguments 'kwargs'. For a list of 'kwargs' see the
    `matplotlib plot documentation <https://tinyurl.com/ycblw9bx>`_

    Parameters
    ----------
    x : :class:`pybamm.Array`
        The array to plot on the x axis
    y : :class:`pybamm.Array`
        The array to plot on the y axis
    ax : matplotlib Axis, optional
        The axis on which to put the plot. If None, a new figure and axis is created.
    testing : bool, optional
        Whether to actually make the plot (turned off for unit tests)
    kwargs
        Keyword arguments, passed to plt.plot

    """
    plt = have_optional_dependency("matplotlib.pyplot")

    if not isinstance(x, pybamm.Array):
        raise TypeError("x must be 'pybamm.Array'")
    if not isinstance(y, pybamm.Array):
        raise TypeError("y must be 'pybamm.Array'")

    if ax is not None:
        testing = True
    else:
        _, ax = plt.subplots()

    ax.plot(x.entries, y.entries, **kwargs)
    ax.set_ylim([ax_min(y.entries), ax_max(y.entries)])

    if not testing:  # pragma: no cover
        plt.show()

    return ax
