#
# Method for creating a dynamic plot
#
import pybamm


def dynamic_plot(*args, **kwargs):
    """
    Creates a :class:`pybamm.QuickPlot` object (with arguments 'args' and keyword
    arguments 'kwargs') and then calls :meth:`pybamm.QuickPlot.dynamic_plot`.
    The key-word argument 'show_plot' is passed to the 'dynamic_plot' method, not the
    `QuickPlot` class.

    Returns
    -------
    plot : :class:`pybamm.QuickPlot`
        The 'QuickPlot' object that was created
    """
    kwargs_for_class = {k: v for k, v in kwargs.items() if k != "show_plot"}
    plot = pybamm.QuickPlot(*args, **kwargs_for_class)
    plot.dynamic_plot(kwargs.get("show_plot", True))
    return plot
