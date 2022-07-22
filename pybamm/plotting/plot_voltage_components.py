#
# Method for plotting voltage components
#
import numpy as np


def plot_voltage_components(
    solution, ax=None, show_legend=True, testing=False, **kwargs_fill
):
    """
    Generate a plot showing the component overpotentials that make up the voltage

    Parameters
    ----------
    solution : :class:`pybamm.Solution`
        Solution object from which to extract voltage components
    ax : matplotlib Axis, optional
        The axis on which to put the plot. If None, a new figure and axis is created.
    show_legend : bool, optional
        Whether to display the legend. Default is True
    testing : bool, optional
        Whether to actually make the plot (turned off for unit tests)
    kwargs_fill
        Keyword arguments, passed to ax.fill_between

    """
    import matplotlib.pyplot as plt

    # Set a default value for alpha, the opacity
    kwargs_fill = {"alpha": 0.6, **kwargs_fill}

    if ax is not None:
        fig = None
        testing = True
    else:
        fig, ax = plt.subplots()

    overpotentials = [
        "X-averaged battery reaction overpotential [V]",
        "X-averaged battery concentration overpotential [V]",
        "X-averaged battery electrolyte ohmic losses [V]",
        "X-averaged battery solid phase ohmic losses [V]",
    ]
    labels = [
        "Reaction overpotential",
        "Concentration overpotential",
        "Ohmic electrolyte overpotential",
        "Ohmic electrode overpotential",
    ]

    # Plot
    # Initialise
    time = solution["Time [h]"].entries
    initial_ocv = solution["X-averaged battery open circuit voltage [V]"](0)
    ocv = solution["X-averaged battery open circuit voltage [V]"].entries
    ax.fill_between(time, ocv, initial_ocv, **kwargs_fill, label="Open-circuit voltage")
    top = ocv
    # Plot components
    for overpotential, label in zip(overpotentials, labels):
        bottom = top + solution[overpotential].entries
        ax.fill_between(time, bottom, top, **kwargs_fill, label=label)
        top = bottom

    V = solution["Battery voltage [V]"].entries
    ax.plot(time, V, "k--", label="Voltage")

    if show_legend:
        leg = ax.legend(loc="lower left", frameon=True)
        leg.get_frame().set_edgecolor("k")

    # Labels
    ax.set_xlim([time[0], time[-1]])
    ax.set_xlabel("Time [h]")

    y_min, y_max = 0.98 * min(np.nanmin(V), np.nanmin(ocv)), 1.02 * (
        max(np.nanmax(V), np.nanmax(ocv))
    )
    ax.set_ylim([y_min, y_max])

    if not testing:  # pragma: no cover
        plt.show()

    return fig, ax
