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

    Pass ``backend="vtk"`` to build a :class:`pybamm.VTKQuickPlot` instead; the
    remaining arguments are then those of that class.

    Returns
    -------
    plot : :class:`pybamm.QuickPlot` or :class:`pybamm.VTKQuickPlot`
        The plot object that was created
    """
    backend = kwargs.pop("backend", "matplotlib")
    show_plot = kwargs.pop("show_plot", True)

    if backend == "vtk":
        plot_class = pybamm.VTKQuickPlot
    elif backend == "matplotlib":
        plot_class = pybamm.QuickPlot
    else:
        raise pybamm.OptionError(
            f"Unknown plotting backend '{backend}'; use 'matplotlib' or 'vtk'."
        )

    plot = plot_class(*args, **kwargs)
    plot.dynamic_plot(show_plot)
    return plot
