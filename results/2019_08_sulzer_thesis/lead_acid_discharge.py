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
    fig.legend(labels, loc="lower center", ncol=len(labels), frameon=True)
    plt.subplots_adjust(bottom=0.25, right=0.95, hspace=1.1, wspace=0.4)
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight")
    else:
        plt.show()


def plot_variables(all_variables, t_eval):
    # Plot
    linestyles = ["k-", "g--", "r:", "b-."]
    file_name = "discharge_electrolyte_comparison.eps"
    n = 3
    m = 3
    fig, axes = plt.subplots(n, m, figsize=(6.4, 6.4))
    labels = [model for model in [x for x in all_variables.values()][0].keys()]
    for k, (Crate, models_variables) in enumerate(all_variables.items()):
        ax = axes.flat[k]
        x_max = models_variables["x [m]"][-1]
        ax.set_xlim([0, x_max])
        # ax.set_ylim([10.5, 13])
        # ax.set_title(
        #     "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
        #         chr(97 + k), Crate, Crate * 0.6
        #     )
        # )

        if k > (m - 1) * n:
            ax.set_xlabel("x [m]")
        if k % m == 0:
            ax.set_ylabel("25% SoC")
        for j, (model, variables) in enumerate(models_variables.items()):
            ax.plot(
                variables["x [m]"](t_eval),
                variables["Electrolyte concentration [Molar]"](t_eval),
                linestyles[j],
            )
    fig.legend(labels, loc="lower center", ncol=len(labels), frameon=True)
    plt.subplots_adjust(bottom=0.25, right=0.95, hspace=1.1, wspace=0.4)
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")
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
    plot_voltages(all_variables, t_eval)
    # Crates = [1, 5]
    # for Crate in Crates:
    #     plot_variables(args.compute, Crate)
