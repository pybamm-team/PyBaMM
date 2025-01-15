#
# Method for plotting voltage components
#
import numpy as np

from pybamm.util import import_optional_dependency
from pybamm.simulation import Simulation
from pybamm.solvers.solution import Solution


def plot_voltage_components(
    input_data,
    ax=None,
    show_legend=True,
    split_by_electrode=False,
    show_plot=True,
    **kwargs_fill,
):
    """
    Generate a plot showing the component overpotentials that make up the voltage

    Parameters
    ----------
    input_data : :class:`pybamm.Solution` or :class:`pybamm.Simulation`
        Solution or Simulation object from which to extract voltage components.
    ax : matplotlib Axis, optional
        The axis on which to put the plot. If None, a new figure and axis is created.
    show_legend : bool, optional
        Whether to display the legend. Default is True
    split_by_electrode : bool, optional
        Whether to show the overpotentials for the negative and positive electrodes
        separately. Default is False.
    show_plot : bool, optional
        Whether to show the plots. Default is True. Set to False if you want to
        only display the plot after plt.show() has been called.
    kwargs_fill
        Keyword arguments: :obj:`matplotlib.axes.Axes.fill_between`
    """
    # Check if the input is a Simulation and extract Solution
    if isinstance(input_data, Simulation):
        solution = input_data.solution
    elif isinstance(input_data, Solution):
        solution = input_data
    plt = import_optional_dependency("matplotlib.pyplot")

    # Set a default value for alpha, the opacity
    kwargs_fill = {"alpha": 0.6, **kwargs_fill}

    if ax is not None:
        fig = None
        show_plot = False
    else:
        fig, ax = plt.subplots(figsize=(8, 4))

    if split_by_electrode is False:
        overpotentials = [
            "Battery particle concentration overpotential [V]",
            "X-averaged battery reaction overpotential [V]",
            "X-averaged battery concentration overpotential [V]",
            "X-averaged battery electrolyte ohmic losses [V]",
            "X-averaged battery solid phase ohmic losses [V]",
        ]
        labels = [
            "Particle concentration overpotential",
            "Reaction overpotential",
            "Electrolyte concentration overpotential",
            "Ohmic electrolyte overpotential",
            "Ohmic electrode overpotential",
        ]
    else:
        overpotentials = [
            "Battery negative particle concentration overpotential [V]",
            "Battery positive particle concentration overpotential [V]",
            "X-averaged battery negative reaction overpotential [V]",
            "X-averaged battery positive reaction overpotential [V]",
            "X-averaged battery concentration overpotential [V]",
            "X-averaged battery electrolyte ohmic losses [V]",
            "X-averaged battery negative solid phase ohmic losses [V]",
            "X-averaged battery positive solid phase ohmic losses [V]",
        ]
        labels = [
            "Negative particle concentration overpotential",
            "Positive particle concentration overpotential",
            "Negative reaction overpotential",
            "Positive reaction overpotential",
            "Electrolyte concentration overpotential",
            "Ohmic electrolyte overpotential",
            "Ohmic negative electrode overpotential",
            "Ohmic positive electrode overpotential",
        ]

    # Plot
    # Initialise
    time = solution["Time [h]"].entries
    if split_by_electrode is False:
        ocv = solution["Battery open-circuit voltage [V]"]
        initial_ocv = ocv(time[0])
        ocv = ocv.entries
        ax.fill_between(
            time, ocv, initial_ocv, **kwargs_fill, label="Open-circuit voltage"
        )
    else:
        ocp_n = solution["Battery negative electrode bulk open-circuit potential [V]"]
        ocp_p = solution["Battery positive electrode bulk open-circuit potential [V]"]
        initial_ocp_n = ocp_n(time[0])
        initial_ocp_p = ocp_p(time[0])
        initial_ocv = initial_ocp_p - initial_ocp_n
        delta_ocp_n = ocp_n.entries - initial_ocp_n
        delta_ocp_p = ocp_p.entries - initial_ocp_p
        ax.fill_between(
            time,
            initial_ocv - delta_ocp_n,
            initial_ocv,
            **kwargs_fill,
            label="Negative open-circuit potential",
        )
        ax.fill_between(
            time,
            initial_ocv - delta_ocp_n + delta_ocp_p,
            initial_ocv - delta_ocp_n,
            **kwargs_fill,
            label="Positive open-circuit potential",
        )
        ocv = initial_ocv - delta_ocp_n + delta_ocp_p
    top = ocv
    # Plot components
    for overpotential, label in zip(overpotentials, labels):
        # negative overpotentials are positive for a discharge and negative for a charge
        # so we have to multiply by -1 to show them correctly
        sgn = -1 if "negative" in overpotential else 1
        bottom = top + sgn * solution[overpotential].entries
        ax.fill_between(time, bottom, top, **kwargs_fill, label=label)
        top = bottom

    V = solution["Battery voltage [V]"].entries
    ax.plot(time, V, "k--", label="Voltage")

    if show_legend:
        leg = ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=True)
        leg.get_frame().set_edgecolor("k")
    if fig is not None:
        fig.tight_layout()

    # Labels
    ax.set_xlim([time[0], time[-1]])
    ax.set_xlabel("Time [h]")

    y_min, y_max = (
        0.98 * min(np.nanmin(V), np.nanmin(ocv)),
        1.02 * (max(np.nanmax(V), np.nanmax(ocv))),
    )
    ax.set_ylim([y_min, y_max])

    if show_plot:  # pragma: no cover
        plt.show()

    return fig, ax
