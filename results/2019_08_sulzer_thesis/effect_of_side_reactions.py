#
# Simulations: effect of side reactions for charge of a lead-acid battery
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
from config import OUTPUT_DIR
from shared_solutions import model_comparison


def plot_voltages(all_variables, t_eval, linestyles, file_name):
    # Plot
    n = int(len(all_variables) // np.sqrt(len(all_variables)))
    m = int(np.ceil(len(all_variables) / n))
    fig, axes = plt.subplots(n, m, figsize=(6, 4))
    labels = [model for model in [x for x in all_variables.values()][0].keys()]
    for k, (Crate, models_variables) in enumerate(all_variables.items()):
        # ax = axes.flat[k]
        ax = axes
        t_max = max(
            np.nanmax(var["Time [h]"](t_eval)) for var in models_variables.values()
        )
        ax.set_xlim([0, t_max])
        ax.set_ylim([12, 15])
        ax.set_xlabel("Time [h]")

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        if k % m == 0:
            ax.set_ylabel("Voltage [V]")
        for j, (model, variables) in enumerate(models_variables.items()):
            ax.plot(
                variables["Time [h]"](t_eval),
                variables["Terminal voltage [V]"](t_eval),
                linestyles[j],
            )
    # ax = plt.subplot(111)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    ax.legend(labels, loc="best")
    fig.tight_layout()
    # plt.subplots_adjust(right=0.2)
    # plt.subplots_adjust(
    #     top=0.92, bottom=0.3, left=0.10, right=0.9, hspace=0.5, wspace=0.5
    # )
    # plt.show()
    plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_interfacial_currents(models_variables, t_eval, file_name):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    output_vars = [
        "Average positive electrode interfacial current density",
        "Average positive electrode oxygen interfacial current density",
        "Average negative electrode oxygen interfacial current density",
        "Average negative electrode interfacial current density",
    ]
    labels = [
        "Positive electrode (main)",
        "Positive electrode (oxygen)",
        "Negative electrode (oxygen)",
        "Negative electrode (main)",
    ]
    t_max = max(np.nanmax(var["Time [h]"](t_eval)) for var in models_variables.values())
    ax.set_xlim([0, t_max])
    # ax.set_ylim([12, 15])
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Interfacial current densities")
    linestyles = ["--", ":", "-.", "-"]
    colors = ["k", "b"]
    for j, (model, variables) in enumerate(models_variables.items()):
        for k, var in enumerate(output_vars):
            ax.plot(
                variables["Time [h]"](t_eval),
                variables[var](t_eval),
                linestyle=linestyles[k],
                color=colors[j],
            )
    ax.legend(labels, loc=2, bbox_to_anchor=(1, 0.7))
    fig.tight_layout()
    # plt.show()
    plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def compare_voltages(all_variables, t_eval):
    # linestyles = ["b:", "k-"]
    # file_name = "effect_of_side_reactions.eps"
    # plot_voltages(all_variables, t_eval, linestyles, file_name)
    file_name = "with_without_reactions_interfacial_currents.eps"
    plot_interfacial_currents(all_variables[-1], t_eval, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    if args.compute:
        pybamm.set_logging_level("INFO")
        models = [
            pybamm.lead_acid.NewmanTiedemann(
                {"side reactions": ["oxygen"]}, name="With oxygen"
            ),
            pybamm.lead_acid.LOQS(name="Without oxygen"),
        ]
        Crates = [-1]
        extra_parameter_values = {
            "Positive electrode"
            + "reference exchange-current density (oxygen) [A.m-2]": 1e-24,
            "Initial State of Charge": 0.5,
        }
        t_eval = np.linspace(0, 2.5, 100)
        all_variables, t_eval = model_comparison(
            models, Crates, t_eval, extra_parameter_values=extra_parameter_values
        )
        with open("effect_of_side_reactions2.pickle", "wb") as f:
            data = (all_variables, t_eval)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    with open("effect_of_side_reactions2.pickle", "rb") as f:
        (all_variables, t_eval) = pickle.load(f)
    compare_voltages(all_variables, t_eval)
