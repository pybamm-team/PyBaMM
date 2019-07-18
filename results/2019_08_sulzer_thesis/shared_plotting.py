#
# Shared plotting
#
import matplotlib.pyplot as plt
import numpy as np
import pybamm

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = None


def plot_variable(
    all_variables, t_eval, times, Crates, variable, file_name, limits_exceptions
):
    linestyles = ["k-", "g--", "r:", "b-."]
    n = len(times)
    m = len(Crates)
    labels = [model for model in [x for x in all_variables.values()][0].keys()]
    x = all_variables[Crates[0]][labels[0]]["x"](0, np.linspace(0, 1))[:, 0]
    x_dim = all_variables[Crates[0]][labels[0]]["x [m]"](0, np.linspace(0, 1))[:, 0]

    fig, axes = plt.subplots(n, m, figsize=(6.4, 6.4))

    # Default limits
    y_min = pybamm.ax_min(
        [
            np.nanmin(variables[variable](t_eval, x))
            for Crate, models_variables in all_variables.items()
            if Crate in Crates
            for variables in models_variables.values()
        ]
    )
    y_max = pybamm.ax_max(
        [
            np.nanmax(variables[variable](t_eval, x))
            for Crate, models_variables in all_variables.items()
            if Crate in Crates
            for variables in models_variables.values()
        ]
    )
    # Exceptions
    if "min" in limits_exceptions:
        y_min = limits_exceptions["min"]
    if "max" in limits_exceptions:
        y_max = limits_exceptions["max"]

    # Plot
    for i, Crate in enumerate(Crates):
        models_variables = all_variables[Crate]
        for j, time in enumerate(times):
            ax = axes[j, i]
            ax.set_xlim([x_dim[0], x_dim[-1]])
            ax.set_ylim([y_min, y_max])
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

            # Title
            if j == 0:
                ax.set_title(
                    "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
                        chr(97 + i), Crate, Crate * 0.6
                    )
                )
            # x-axis
            if j == len(times) - 1:
                ax.set_xlabel("x [m]")
            else:
                ax.set_xticklabels([])

            # y-axis
            if i == 0:
                soc = models_variables["LOQS"]["State of Charge"](time)
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
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight")
