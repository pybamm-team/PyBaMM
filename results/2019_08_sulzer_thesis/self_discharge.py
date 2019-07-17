#
# Simulations: self-discharge
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
    ax.legend(labels, loc="best")
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def compare_voltages(all_variables, t_eval):
    linestyles = ["b:", "k-", "r--"]
    file_name = "self_discharge.eps"
    plot_voltages(all_variables, t_eval, linestyles, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    if args.compute:
        pybamm.set_logging_level("INFO")
        models = [
            pybamm.lead_acid.NewmanTiedemann(name="Full, without oxygen"),
            pybamm.lead_acid.NewmanTiedemann(
                {"side reactions": ["oxygen"]}, name="Full, with oxygen"
            ),
            pybamm.lead_acid.LOQS(
                {"surface form": "algebraic", "side reactions": ["oxygen"]},
                name="Leading-order, with side oxygen",
            ),
        ]
        Crates = [1]
        extra_parameter_values = {
            "Current function": pybamm.GetConstantCurrent(current=0),
            "Positive electrode"
            + "reference exchange-current density (oxygen) [A.m-2]": 1e-22,
            "Initial State of Charge": 1,
        }
        t_eval = np.linspace(0, 1000, 100)
        all_variables, t_eval = model_comparison(
            models, Crates, t_eval, extra_parameter_values=extra_parameter_values
        )
        with open("self_discharge.pickle", "wb") as f:
            data = (all_variables, t_eval)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    with open("self_discharge.pickle", "rb") as f:
        (all_variables, t_eval) = pickle.load(f)
    compare_voltages(all_variables, t_eval)
