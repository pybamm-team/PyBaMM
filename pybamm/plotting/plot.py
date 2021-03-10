#
# Method for creating a 1D plot of pybamm arrays
#
import pybamm
from .quick_plot import ax_min, ax_max


def plot(x, y, xlabel=None, ylabel=None, title=None, testing=False, **kwargs):
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
    xlabel : str, optional
        The label for the x axis
    ylabel : str, optional
        The label for the y axis
    testing : bool, optional
        Whether to actually make the plot (turned off for unit tests)
    kwargs
        Keyword arguments, passed to plt.plot

    """
    import matplotlib.pyplot as plt

    if not isinstance(x, pybamm.Array):
        raise TypeError("x must be 'pybamm.Array'")
    if not isinstance(y, pybamm.Array):
        raise TypeError("y must be 'pybamm.Array'")

    plt.plot(x.entries, y.entries, **kwargs)
    plt.ylim([ax_min(y.entries), ax_max(y.entries)])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if not testing:  # pragma: no cover
        plt.show()

    return
