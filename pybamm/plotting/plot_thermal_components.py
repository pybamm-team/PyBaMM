#
# Method for plotting voltage components
#

from pybamm.util import import_optional_dependency
from pybamm.simulation import Simulation
from pybamm.solvers.solution import Solution
from scipy.integrate import cumulative_trapezoid


def plot_thermal_components(
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
    show_plot : bool, optional
        Whether to show the plots. Default is True. Set to False if you want to
        only display the plot after plt.show() has been called.
    kwargs_fill
        Keyword arguments, passed to ax.fill_between

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
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    time_s = solution["Time [s]"].entries
    time_h = time_s / 3600
    volume = solution["Cell thermal volume [m3]"].entries

    heating_sources = [
        "Lumped total cooling",
        "Ohmic heating",
        "Irreversible electrochemical heating",
        "Reversible heating",
    ]
    try:
        heats = {
            name: solution[name + " [W]"].entries / volume for name in heating_sources
        }
    except KeyError as err:
        raise NotImplementedError(
            "plot_thermal_components is only implemented for lumped models"
        ) from err

    cumul_heats = {
        name: cumulative_trapezoid(heat, time_s, initial=0)
        for name, heat in heats.items()
    }

    # Plot
    # Initialise
    total_heat = 0
    bottom_heat = heats["Lumped total cooling"]
    total_cumul_heat = 0
    bottom_cumul_heat = cumul_heats["Lumped total cooling"]
    # Plot components
    for name in heating_sources:
        top_heat = bottom_heat + abs(heats[name])
        ax[0].fill_between(time_h, bottom_heat, top_heat, **kwargs_fill, label=name)
        bottom_heat = top_heat
        total_heat += heats[name]

        top_cumul_heat = bottom_cumul_heat + abs(cumul_heats[name])
        ax[1].fill_between(
            time_h, bottom_cumul_heat, top_cumul_heat, **kwargs_fill, label=name
        )
        bottom_cumul_heat = top_cumul_heat
        total_cumul_heat += cumul_heats[name]

    ax[0].plot(time_h, total_heat, "k--")
    ax[1].plot(time_h, total_cumul_heat, "k--", label="Total")

    if show_legend:
        leg = ax[1].legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=True)
        leg.get_frame().set_edgecolor("k")

    # Labels
    for a in ax:
        a.set_xlabel("Time [h]")
        a.set_xlim([time_h[0], time_h[-1]])

    ax[0].set_title("Heat generation [W/m$^3$]")
    ax[1].set_title("Cumulative heat generation [J/m$^3$]")

    if fig is not None:
        fig.tight_layout()

    if show_plot:  # pragma: no cover
        plt.show()

    return fig, ax
