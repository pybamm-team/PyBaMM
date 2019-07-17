#
# Simulations: discharge of a lead-acid battery
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
from shared import model_comparison, simulation

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = None


def plot_voltages(all_variables, t_eval):
    # Plot
    linestyles = ["k-", "g--", "r:", "b-."]
    file_name = "discharge_voltage_comparison.eps"
    n = int(len(all_variables) // np.sqrt(len(all_variables)))
    m = int(np.ceil(len(all_variables) / n))
    fig, axes = plt.subplots(n, m, figsize=(6.4, 4.5))
    labels = [model for model in [x for x in all_variables.values()][0].keys()]
    for k, (Crate, models_variables) in enumerate(all_variables.items()):
        ax = axes.flat[k]
        t_max = max(
            np.nanmax(var["Time [h]"](t_eval)) for var in models_variables.values()
        )
        ax.set_xlim([0, t_max])
        ax.set_ylim([10.5, 13])
        ax.set_xlabel("Time [h]")
        ax.set_title(
            "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
                chr(97 + k), Crate, Crate * 0.6
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
    leg = fig.legend(labels, loc="lower center", ncol=len(labels), frameon=True)
    leg.get_frame().set_edgecolor("k")
    plt.subplots_adjust(bottom=0.25, right=0.95, hspace=1.1, wspace=0.4)
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight")


def plot_variables(all_variables, t_eval):
    # Set up
    Crates = [0.1, 2, 5]
    times = np.linspace(0, 0.5, 4)
    linestyles = ["k-", "g--", "r:", "b-."]
    var_file_names = {
        "Electrolyte concentration [Molar]": "discharge_electrolyte_concentration_comparison.eps",
        "Electrolyte potential [V]": "discharge_electrolyte_potential_comparison.eps",
        "Interfacial current density": "discharge_interfacial_current_density_comparison.eps",
    }
    limits_exceptions = {"Electrolyte concentration [Molar]": {"min": 0}}
    n = len(times)
    m = len(Crates)
    labels = [model for model in [x for x in all_variables.values()][0].keys()]
    x = all_variables[Crates[0]][labels[0]]["x"](0, np.linspace(0, 1))[:, 0]
    x_dim = all_variables[Crates[0]][labels[0]]["x [m]"](0, np.linspace(0, 1))[:, 0]
    for var, file_name in var_file_names.items():
        fig, axes = plt.subplots(n, m, figsize=(6.4, 6.4))
        y_min = pybamm.ax_min(
            [
                np.nanmin(variables[var](t_eval, x))
                for models_variables in all_variables.values()
                for variables in models_variables.values()
            ]
        )
        y_max = pybamm.ax_max(
            [
                np.nanmax(variables[var](t_eval, x))
                for models_variables in all_variables.values()
                for variables in models_variables.values()
            ]
        )
        if var in limits_exceptions:
            exceptions = limits_exceptions[var]
            if "min" in exceptions:
                y_min = exceptions["min"]
            if "max" in exceptions:
                y_max = exceptions["max"]
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
                    ax.plot(x_dim, variables[var](time, x), linestyles[j])
        fig.legend(labels, loc="lower center", ncol=len(labels), frameon=True)
        plt.subplots_adjust(
            bottom=0.15, top=0.95, left=0.18, right=0.97, hspace=0.08, wspace=0.05
        )
        if OUTPUT_DIR is not None:
            plt.savefig(
                OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight"
            )


def lead_acid_discharge():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    if args.compute:
        models = [
            pybamm.lead_acid.NewmanTiedemann(name="Full"),
            pybamm.lead_acid.LOQS(name="LOQS"),
            pybamm.lead_acid.FOQS(name="FOQS"),
            pybamm.lead_acid.Composite(name="Composite"),
        ]
        Crates = [0.1, 0.2, 0.5, 1, 2, 5]
        t_eval = np.linspace(0, 1, 100)
        extra_parameter_values = {"Bruggeman coefficient": 0.001}
        all_variables, t_eval = model_comparison(
            models, Crates, t_eval, extra_parameter_values=extra_parameter_values
        )
        with open("discharge_asymptotics_data.pickle", "wb") as f:
            data = (all_variables, t_eval)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open("discharge_asymptotics_data.pickle", "rb") as f:
            (all_variables, t_eval) = pickle.load(f)
    # plot_voltages(all_variables, t_eval)
    plot_variables(all_variables, t_eval)


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")
    lead_acid_discharge()
    plt.show()
