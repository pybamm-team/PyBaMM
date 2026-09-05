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

    Pass ``backend="vtk"`` to use the VTK-based viewer for unstructured
    mesh solutions instead of matplotlib.

    Returns
    -------
    plot : :class:`pybamm.QuickPlot` or :class:`pybamm.VTKQuickPlot`
        The plot object that was created
    """
    backend = kwargs.pop("backend", "matplotlib")
    show_plot = kwargs.pop("show_plot", True)

    if backend == "vtk":
        from pybamm.plotting.plot_vtk import VTKQuickPlot

        output_variables = kwargs.pop("output_variables", None)
        options = kwargs.pop("options", None)
        interpolate_time = kwargs.pop("interpolate_time", False)
        plot = VTKQuickPlot(
            *args,
            output_variables=output_variables,
            options=options,
            interpolate_time=interpolate_time,
            **kwargs,
        )
        plot.dynamic_plot(show_plot)
        return plot

    kwargs_for_class = {k: v for k, v in kwargs.items()}
    plot = pybamm.QuickPlot(*args, **kwargs_for_class)
    plot.dynamic_plot(show_plot)
    return plot
