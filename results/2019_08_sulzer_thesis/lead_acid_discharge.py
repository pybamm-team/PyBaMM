#
# Simulations: discharge of a lead-acid battery
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
from config import OUTPUT_DIR
from shared import model_comparison


def plot_voltages(all_variables, t_eval, models, linestyles, file_name):
    # Plot
    n = int(len(all_variables) // np.sqrt(len(all_variables)))
    m = int(np.ceil(len(all_variables) / n))
    fig, axes = plt.subplots(n, m, figsize=(8, 4.5))
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
            "\\textbf{{{}C}} ($\\mathcal{{C}}_e={}$)".format(Crate, Crate * 0.6)
        )
        # ax.set_title(
        #     "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
        #         chr(97 + k), Crate, Crate * 0.6
        #     )
        # )

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        if k % m == 0:
            ax.set_ylabel("Voltage [V]")
        for j, (model, variables) in enumerate(models_variables.items()):
            if model in models:
                ax.plot(
                    variables["Time [h]"](t_eval),
                    variables["Terminal voltage [V]"](t_eval) * 6,
                    linestyles[j],
                )
    ax.legend(labels, bbox_to_anchor=(1.05, 2), loc=2)
    fig.tight_layout()
    # plt.show()
    plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_variables(all_variables, t_eval, variables, models, linestyles, file_name):
    plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)
    plot.plot(0.5)
    plot.tight_layout()
    plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def compare_voltages_composite(all_variables, t_eval):
    models = ["Full", "Leading-order", "First-order", "Composite"]
    linestyles = ["k-", "g--", "r:", "b-."]
    file_name = "discharge_voltage_comparison.eps"
    plot_voltages(all_variables, t_eval, models, linestyles, file_name)


def compare_voltages_quasistatic(all_variables, t_eval):
    models = ["Full", "Leading-order", "First-order"]
    linestyles = ["k-", "g--", "r:"]
    file_name = "discharge_voltage_comparison_quasistatic.eps"
    plot_voltages(all_variables, t_eval, models, linestyles, file_name)


def plot_states(all_variables, t_eval):
    mesh = 1
    solutions = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    if args.compute:
        pybamm.set_logging_level("INFO")
        models = [
            pybamm.lead_acid.NewmanTiedemann(name="Full"),
            pybamm.lead_acid.LOQS(name="Leading-order"),
            pybamm.lead_acid.FOQS(name="First-order"),
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
    with open("discharge_asymptotics_data.pickle", "rb") as f:
        (all_variables, t_eval) = pickle.load(f)
    compare_voltages_composite(all_variables, t_eval)
    compare_voltages_quasistatic(all_variables, t_eval)
