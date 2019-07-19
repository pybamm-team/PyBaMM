#
# Shared plotting
#
import matplotlib.pyplot as plt
import numpy as np
import pybamm


def plot_voltages(all_variables, t_eval):
    # Plot
    linestyles = ["k-", "g--", "r:", "b-."]
    n = int(len(all_variables) // np.sqrt(len(all_variables)))
    m = int(np.ceil(len(all_variables) / n))
    fig, axes = plt.subplots(n, m, figsize=(6.4, 4.5))
    labels = [model for model in [x for x in all_variables.values()][0].keys()]
    y_min = 0.98 * min(
        np.nanmin(variables["Terminal voltage [V]"](t_eval))
        for models_variables in all_variables.values()
        for variables in models_variables.values()
    )
    y_max = 1.02 * max(
        np.nanmax(variables["Terminal voltage [V]"](t_eval))
        for models_variables in all_variables.values()
        for variables in models_variables.values()
    )
    # Strict voltage cut-offs
    y_min = max(y_min, 10.5)
    y_max = min(y_max, 14.5)
    for k, (Crate, models_variables) in enumerate(all_variables.items()):
        if len(all_variables) == 1:
            ax = axes
        else:
            ax = axes.flat[k]
        t_max = max(
            np.nanmax(var["Time [h]"](t_eval)) for var in models_variables.values()
        )
        ax.set_xlim([0, t_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Time [h]")
        if len(all_variables) > 1:
            ax.set_title(
                "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
                    chr(97 + k), abs(Crate), abs(Crate) * 0.6
                )
            )
            # Hide the right and top spines
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            # Only show ticks on the left and bottom spines
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        if k % m == 0:
            ax.set_ylabel("Voltage [V]")
        for j, (model, variables) in enumerate(models_variables.items()):
            ax.plot(
                variables["Time [h]"](t_eval),
                variables["Terminal voltage [V]"](t_eval),
                linestyles[j],
            )
    if len(all_variables) == 1:
        leg = ax.legend(labels, loc="best")
        fig.tight_layout()
    else:
        leg = fig.legend(labels, loc="lower center", ncol=len(labels))
        plt.subplots_adjust(bottom=0.25, right=0.95, hspace=1.1, wspace=0.4)
    leg.get_frame().set_edgecolor("k")


def plot_variable(all_variables, times, variable, limits_exceptions=None):
    linestyles = ["k-", "g--", "r:", "b-."]
    n = len(times)
    m = len(all_variables)
    Crates = list(all_variables.keys())
    labels = [model for model in [x for x in all_variables.values()][0].keys()]
    x = all_variables[Crates[0]][labels[0]]["x"](0, np.linspace(0, 1))[:, 0]
    x_dim = all_variables[Crates[0]][labels[0]]["x [m]"](0, np.linspace(0, 1))[:, 0]

    fig, axes = plt.subplots(n, m, figsize=(6.4, 6.4))

    # Default limits
    y_min = pybamm.ax_min(
        [
            np.nanmin(variables[variable](times, x))
            for Crate, models_variables in all_variables.items()
            for variables in models_variables.values()
        ]
    )
    y_max = pybamm.ax_max(
        [
            np.nanmax(variables[variable](times, x))
            for Crate, models_variables in all_variables.items()
            for variables in models_variables.values()
        ]
    )
    # Exceptions
    if "min" in limits_exceptions:
        y_min = limits_exceptions["min"]
    if "max" in limits_exceptions:
        y_max = limits_exceptions["max"]

    # Plot
    for i, (Crate, models_variables) in enumerate(all_variables.items()):
        for j, time in enumerate(times):
            ax = axes[j, i]
            ax.set_xlim([x_dim[0], x_dim[-1]])
            ax.set_ylim([y_min, y_max])
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

            # Title
            if j == 0:
                ax.set_title(
                    "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
                        chr(97 + i), abs(Crate), abs(Crate) * 0.6
                    )
                )
            # x-axis
            if j == len(times) - 1:
                ax.set_xlabel("x [m]")
            else:
                ax.set_xticklabels([])

            # y-axis
            if i == 0:
                soc = [x for x in models_variables.values()][0]["State of Charge"](time)
                ax.set_ylabel("{}\% SoC".format(int(soc)), rotation=0, labelpad=30)
                ax.yaxis.get_label().set_verticalalignment("center")
            else:
                ax.set_yticklabels([])

            # Plot
            for j, (model, variables) in enumerate(models_variables.items()):
                ax.plot(x_dim, variables[variable](time, x), linestyles[j])
    leg = fig.legend(labels, loc="lower center", ncol=len(labels), frameon=True)
    leg.get_frame().set_edgecolor("k")
    plt.subplots_adjust(
        bottom=0.15, top=0.95, left=0.18, right=0.97, hspace=0.08, wspace=0.05
    )


def plot_voltage_breakdown(all_variables, t_eval, model, Crates):
    n = int(len(Crates) // np.sqrt(len(Crates)))
    m = int(np.ceil(len(Crates) / n))
    fig, axes = plt.subplots(n, m, figsize=(6.4, 2.3))
    labels = ["V", "$V_U$", "$V_k$", "$V_c$", "$V_o$"]
    overpotentials = [
        "Average reaction overpotential [V]",
        "Average concentration overpotential [V]",
        "Average electrolyte ohmic losses [V]",
    ]
    y_min = 0.95 * min(
        np.nanmin(models_variables[model]["Terminal voltage [V]"](t_eval))
        for models_variables in all_variables.values()
    )
    y_max = 1.05 * max(
        np.nanmax(models_variables[model]["Terminal voltage [V]"](t_eval))
        for models_variables in all_variables.values()
    )
    for k, Crate in enumerate(Crates):
        variables = all_variables[Crate][model]
        ax = axes.flat[k]

        # Set up
        t_max = np.nanmax(variables["Time [h]"](t_eval))
        ax.set_xlim([0, t_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Time [h]")
        ax.set_title(
            "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
                chr(97 + k), abs(Crate), abs(Crate) * 0.6
            )
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        if k % m == 0:
            ax.set_ylabel("Voltage [V]")

        # Plot
        # Initialise
        time = variables["Time [h]"](t_eval)
        initial_ocv = variables["Average open circuit voltage [V]"](0) * 6
        ocv = variables["Average open circuit voltage [V]"](t_eval) * 6
        ax.fill_between(time, ocv, initial_ocv)
        top = ocv
        # Plot
        for j, overpotential in enumerate(overpotentials):
            bottom = top + variables[overpotential](t_eval) * 6
            ax.fill_between(time, bottom, top)
            top = bottom
        ax.plot(time, variables["Terminal voltage [V]"](t_eval), "k--")
    leg = axes.flat[-1].legend(
        labels, bbox_to_anchor=(1.05, 0.5), loc="center left", frameon=True
    )
    leg.get_frame().set_edgecolor("k")
    fig.tight_layout()
