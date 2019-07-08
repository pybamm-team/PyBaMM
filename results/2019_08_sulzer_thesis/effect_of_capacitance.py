#
# Simulations: discharge of a lead-acid battery
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
from config import OUTPUT_DIR
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shared import model_comparison


def plot_voltages(all_variables, t_eval, Crates):
    # Only use some Crates
    all_variables = {k: v for k, v in all_variables.items() if k in Crates}
    # Plot
    plt.subplots()
    n = int(len(all_variables) // np.sqrt(len(all_variables)))
    m = np.ceil(len(all_variables) / n)
    for k, (Crate, models_variables) in enumerate(all_variables.items()):
        t_max = max(
            np.nanmax(var["Time [s]"](t_eval)) for var in models_variables.values()
        )
        ax = plt.subplot(n, m, k + 1)
        plt.axis([0, t_max, 10.5, 13])
        ax.set_xlabel("Time [s]")
        if len(Crates) > 1:
            plt.title("\\textbf{{({})}} {} C".format(chr(97 + k), Crate), y=-0.4)

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

        # Add inset plot
        inset = inset_axes(ax, width="30%", height="30%", loc=3, borderpad=3)

        # Linestyles
        linestyles = ["k-", "b-.", "r--"]
        for j, (model_name, variables) in enumerate(models_variables.items()):
            if k == 0:
                label = model_name
            else:
                label = None
            if k % m == 0:
                ax.set_ylabel("Voltage [V]")

            ax.plot(
                variables["Time [s]"](t_eval),
                variables["Terminal voltage [V]"](t_eval) * 6,
                linestyles[j],
                label=label,
            )
            inset.plot(
                variables["Time [s]"](t_eval[:40]),
                variables["Terminal voltage [V]"](t_eval[:40]) * 6,
                linestyles[j],
            )
        # plt.legend(loc="upper right")
    file_name = "capacitance_voltage_comparison.eps".format(Crate)
    plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_errors(all_variables, t_eval, Crates):
    def rmse(predictions, targets):
        return np.sqrt(np.nanmean((predictions - targets) ** 2))

    # Only use some Crates
    all_variables = {k: v for k, v in all_variables.items() if k in Crates}
    # Plot
    plt.subplots()
    for k, (Crate, models_variables) in enumerate(all_variables.items()):
        ax = plt.subplot(1, 1, 1)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error [V]")

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

        # Linestyles
        linestyles = ["k-", "b-.", "r--"]

        for j, (model, variables) in enumerate(models_variables.items()):
            options = dict(model[1])
            if options["surface form"] is False:
                base_model_results = models_variables[model]
                continue
            if k == 0:
                label = model[0]
            else:
                label = None

            error = np.abs(
                variables["Terminal voltage [V]"](t_eval)
                - base_model_results["Terminal voltage [V]"](t_eval)
            )
            ax.loglog(variables["Time [s]"](t_eval), error, linestyles[j], label=label)
        # plt.legend(loc="upper right")
    file_name = "capacitance_errors_voltages.eps".format(Crate)
    plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    t_eval = np.concatenate([np.logspace(-6, -3, 50), np.linspace(0.001, 1, 100)])
    if args.compute:
        pybamm.set_logging_level("INFO")
        models = [
            pybamm.lead_acid.NewmanTiedemann(),
            pybamm.lead_acid.NewmanTiedemann({"surface form": "differential"}),
            pybamm.lead_acid.NewmanTiedemann({"surface form": "algebraic"}),
        ]
        Crates = [0.1, 0.5, 1, 2]
        all_variables, t_eval = model_comparison(models, Crates, t_eval)
        with open("capacitance_data.pickle", "wb") as f:
            data = (all_variables, t_eval)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    with open("capacitance_data.pickle", "rb") as f:
        (all_variables, t_eval) = pickle.load(f)
    plot_voltages(all_variables, t_eval, [1])
    plot_errors(all_variables, t_eval, [0.1, 0.5, 1, 2])
