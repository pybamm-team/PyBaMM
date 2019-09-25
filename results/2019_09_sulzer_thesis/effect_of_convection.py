#
# Effect of convection for discharge of a lead-acid battery
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


def plot_voltages(all_variables, t_eval, operation="standard"):
    linestyles = ["k-", "r--"]
    shared_plotting.plot_voltages(all_variables, t_eval, linestyles, figsize=(6.4, 2.5))
    if operation == "big beta":
        file_name = "convection_voltage_comparison_bigger_beta.eps"
    elif operation == "high C":
        file_name = "convection_voltage_comparison_high_C.eps"
    else:
        file_name = "convection_voltage_comparison.eps"
    plt.subplots_adjust(bottom=0.4)
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_variables(all_variables, t_eval, operation="standard"):
    # Set up
    times = np.array([0.195])
    linestyles = ["k-", "r--"]
    if operation == "big beta":
        var_file_names = {
            "Volume-averaged velocity [m.s-1]"
            + "": "convection_velocity_comparison_bigger_beta.eps",
            "Electrolyte concentration [Molar]"
            + "": "convection_electrolyte_concentration_comparison_bigger_beta.eps",
        }
    elif operation == "high C":
        var_file_names = {
            "Volume-averaged velocity [m.s-1]"
            + "": "convection_velocity_comparison_high_C.eps",
            "Electrolyte concentration [Molar]"
            + "": "convection_electrolyte_concentration_comparison_high_C.eps",
        }
    elif operation == "standard":
        var_file_names = {
            "Volume-averaged velocity [m.s-1]": "convection_velocity_comparison.eps",
            "Electrolyte concentration [Molar]"
            + "": "convection_electrolyte_concentration_comparison.eps",
        }
    for var, file_name in var_file_names.items():
        fig, axes = shared_plotting.plot_variable(
            all_variables, times, var, linestyles=linestyles, figsize=(6.4, 3)
        )
        for ax in axes.flat:
            title = ax.get_title()
            ax.set_title(title, y=1.08)
        plt.subplots_adjust(
            bottom=0.3, top=0.85, left=0.1, right=0.9, hspace=0.08, wspace=0.05
        )
        if OUTPUT_DIR is not None:
            plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def charge_states(compute):
    savefile = "effect_of_convection_data.pickle"
    if compute:
        models = [
            pybamm.lead_acid.Full(
                {"convection": {"transverse": "uniform"}}, name="With convection"
            ),
            pybamm.lead_acid.Full(name="Without convection"),
        ]
        Crates = [0.5, 1, 5]
        t_eval = np.linspace(0, 1, 100)
        all_variables, t_eval = model_comparison(models, Crates, t_eval)
        with open(savefile, "wb") as f:
            data = (all_variables, t_eval)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(savefile, "rb") as f:
                (all_variables, t_eval) = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute' first to generate results"
            )
    plot_voltages(all_variables, t_eval)
    plot_variables(all_variables, t_eval)


def charge_states_high_C(compute):
    savefile = "effect_of_convection_high_C_data.pickle"
    if compute:
        models = [
            pybamm.lead_acid.Full(
                {"surface form": "algebraic", "convection": {"transverse": "uniform"}},
                name="With convection",
            ),
            pybamm.lead_acid.Full(
                {"surface form": "algebraic"}, name="Without convection"
            ),
        ]
        Crates = [10, 25]
        t_eval = np.linspace(0, 0.24, 100)
        all_variables, t_eval = model_comparison(models, Crates, t_eval)
        with open(savefile, "wb") as f:
            data = (all_variables, t_eval)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(savefile, "rb") as f:
                (all_variables, t_eval) = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute' first to generate results"
            )
    plot_voltages(all_variables, t_eval, operation="high C")
    plot_variables(all_variables, t_eval, operation="high C")


def charge_states_bigger_volume_change(compute):
    savefile = "effect_of_convection_bigger_beta_data.pickle"
    if compute:
        models = [
            pybamm.lead_acid.Full(
                {"convection": {"transverse": "uniform"}}, name="With convection"
            ),
            pybamm.lead_acid.Full(name="Without convection"),
        ]
        Crates = [0.5, 1, 5]
        t_eval = np.linspace(0, 1, 100)
        extra_parameter_values = {"Volume change factor": 5}
        all_variables, t_eval = model_comparison(
            models, Crates, t_eval, extra_parameter_values=extra_parameter_values
        )
        with open(savefile, "wb") as f:
            data = (all_variables, t_eval)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(savefile, "rb") as f:
                (all_variables, t_eval) = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute' first to generate results"
            )
    plot_voltages(all_variables, t_eval, operation="big beta")
    plot_variables(all_variables, t_eval, operation="big beta")


if __name__ == "__main__":
    pybamm.set_logging_level("DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    charge_states(args.compute)
    charge_states_high_C(args.compute)
    charge_states_bigger_volume_change(args.compute)
    plt.show()
