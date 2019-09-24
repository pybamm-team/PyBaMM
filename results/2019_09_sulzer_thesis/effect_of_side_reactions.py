#
# Effect of side reactions for charge of a lead-acid battery
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
import shared_plotting
from shared_solutions import model_comparison

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = None


def plot_voltages(all_variables, t_eval):
    linestyles = ["k-", "r--"]
    shared_plotting.plot_voltages(all_variables, t_eval, linestyles, figsize=(6.4, 2.5))
    file_name = "side_reactions_voltage_comparison.eps"
    plt.subplots_adjust(bottom=0.4)
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_interfacial_currents(all_variables, t_eval):
    file_name = "side_reactions_interfacial_current_density_comparison.eps"
    output_vars = [
        "X-averaged positive electrode interfacial current density",
        "X-averaged positive electrode oxygen interfacial current density",
        "X-averaged negative electrode oxygen interfacial current density",
        "X-averaged negative electrode interfacial current density",
    ]
    labels = [
        "Pos electrode\n(main)",
        "Pos electrode\n(oxygen)",
        "Neg electrode\n(oxygen)",
        "Neg electrode\n(main)",
    ]
    shared_plotting.plot_time_dependent_variables(
        all_variables, t_eval, output_vars, labels
    )
    plt.subplots_adjust(bottom=0.4, right=0.95, wspace=0.3)
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def charge_states(compute):
    savefile1 = "effect_of_side_reactions_data.pickle"
    savefile2 = "effect_of_side_reactions_loqs_data.pickle"
    if compute:
        models1 = [
            pybamm.lead_acid.Full(
                {"surface form": "algebraic", "side reactions": ["oxygen"]},
                name="With oxygen",
            ),
            pybamm.lead_acid.Full(
                {"surface form": "algebraic"}, name="Without oxygen"
            ),
        ]
        Crates = [-0.1, -1, -5]
        t_eval = np.linspace(0, 4.5, 100)
        extra_parameter_values = {"Initial State of Charge": 0.5}
        all_variables1, t_eval1 = model_comparison(
            models1, Crates, t_eval, extra_parameter_values=extra_parameter_values
        )
        # Use LOQS without voltage cut-off for interfacial current densities, so that
        # the current goes all the way
        models2 = [
            pybamm.lead_acid.Full(
                {"surface form": "algebraic", "side reactions": ["oxygen"]},
                name="With oxygen",
            ),
            pybamm.lead_acid.LOQS({"surface form": "algebraic"}, name="Without oxygen"),
        ]
        extra_parameter_values["Upper voltage cut-off [V]"] = 100
        all_variables2, t_eval2 = model_comparison(
            models2, Crates, t_eval, extra_parameter_values=extra_parameter_values
        )
        with open(savefile1, "wb") as f:
            data1 = (all_variables1, t_eval1)
            pickle.dump(data1, f, pickle.HIGHEST_PROTOCOL)
        with open(savefile2, "wb") as f:
            data2 = (all_variables2, t_eval2)
            pickle.dump(data2, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(savefile1, "rb") as f:
                (all_variables1, t_eval1) = pickle.load(f)
            with open(savefile2, "rb") as f:
                (all_variables2, t_eval2) = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute' first to generate results"
            )
    plot_voltages(all_variables1, t_eval1)
    plot_interfacial_currents(all_variables2, t_eval2)


if __name__ == "__main__":
    pybamm.set_logging_level("DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    charge_states(args.compute)
    plt.show()
