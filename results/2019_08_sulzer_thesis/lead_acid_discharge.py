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


def plot_voltages(all_variables, t_eval):
    # Plot
    plt.subplots(figsize=(6, 4))
    n = int(len(all_variables) // np.sqrt(len(all_variables)))
    m = np.ceil(len(all_variables) / n)
    for k, Crate in enumerate(all_variables.keys()):
        models_variables = all_variables[Crate]
        t_max = max(
            np.nanmax(var["Time [h]"](t_eval)) for var in models_variables.values()
        )
        ax = plt.subplot(n, m, k + 1)
        plt.axis([0, t_max, 10.5, 13])
        plt.xlabel("Time [h]")
        plt.title("\\textbf{{({})}} {}C".format(chr(97 + k), Crate))

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        for model, variables in models_variables.items():
            if k == 0:
                label = model[0]
            else:
                label = None
            if k % m == 0:
                plt.ylabel("Voltage [V]")

            plt.plot(
                variables["Time [h]"](t_eval),
                variables["Terminal voltage [V]"](t_eval) * 6,
                label=label,
            )
        # plt.legend(loc="upper right")
    file_name = "discharge_voltage_comparison.eps".format(Crate)
    plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    t_eval = np.linspace(0, 1, 100)
    if args.compute:
        pybamm.set_logging_level("INFO")
        models = [
            pybamm.lead_acid.LOQS(),
            pybamm.lead_acid.Composite(),
            pybamm.lead_acid.NewmanTiedemann(),
        ]
        Crates = [0.1, 0.2, 0.5, 1, 2, 5]
        all_variables, t_eval = model_comparison(models, Crates, t_eval)
        with open("discharge_asymptotics_data.pickle", "wb") as f:
            data = (all_variables, t_eval)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    with open("discharge_asymptotics_data.pickle", "rb") as f:
        (all_variables, t_eval) = pickle.load(f)
    plot_voltages(all_variables, t_eval)
