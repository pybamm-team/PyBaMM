#
# Simulations: discharge of a lead-acid battery
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
import shared_plotting
from config import OUTPUT_DIR
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shared_solutions import model_comparison, convergence_study

save_folder = "results/2019_08_sulzer_thesis/data/capacitance_results/"


def plot_voltages(all_variables, t_eval):
    linestyles = ["k-", "b-.", "r--"]
    _, axes = shared_plotting.plot_voltages(
        all_variables, t_eval, linestyles=linestyles
    )

    # Add inset plot
    for k, (Crate, models_variables) in enumerate(all_variables.items()):
        ax = axes.flat[k]
        y_min = ax.get_ylim()[0]
        ax.set_ylim([y_min, 13.6])
        inset = inset_axes(ax, width="40%", height="40%", loc=1, borderpad=0)
        for j, variables in enumerate(models_variables.values()):
            time = variables["Time [s]"](t_eval)
            capacitance_indices = np.where(time < 50)
            time = time[capacitance_indices]
            voltage = variables["Battery voltage [V]"](t_eval)[capacitance_indices]
            inset.plot(time, voltage, linestyles[j])
            inset.set_xlabel("Time [s]", fontsize=10)
            inset.set_xlim([0, 3])

    file_name = "capacitance_voltage_comparison.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight")


def plot_errors(all_variables, t_eval, Crates):
    # Linestyles
    linestyles = ["k-", "b-.", "r--"]
    # Only use some Crates
    all_variables = {k: v for k, v in all_variables.items() if k in Crates}
    # Plot
    fig, ax = plt.subplots(1, 1)
    for k, (Crate, models_variables) in enumerate(all_variables.items()):
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Error [V]")

        for j, (model, variables) in enumerate(models_variables.items()):
            if model == "direct form":
                base_model_results = models_variables[model]
                continue
            error = np.abs(
                variables["Battery voltage [V]"](t_eval)
                - base_model_results["Battery voltage [V]"](t_eval)
            )
            ax.loglog(variables["Time [h]"](t_eval), error, linestyles[j], label=model)
        ax.legend(loc="best")
    fig.tight_layout()
    file_name = "capacitance_errors_voltages.eps".format(Crate)
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def discharge_states(compute):
    savefile = "effect_of_capacitance_data.pickle"
    if compute:
        models = [
            pybamm.lead_acid.NewmanTiedemann(name="direct form"),
            pybamm.lead_acid.NewmanTiedemann(
                {"surface form": "differential"},
                name="capacitance form\n(differential)",
            ),
            pybamm.lead_acid.NewmanTiedemann(
                {"surface form": "algebraic"}, name="capacitance form\n(algebraic)"
            ),
        ]
        Crates = [0.1, 1, 5]
        t_eval = np.concatenate(
            [np.logspace(-6, -3, 50), np.linspace(0.001, 1, 100)[1:]]
        )
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
    plot_errors(all_variables, t_eval, [5])


def plot_times(models_times_and_voltages):
    shared_plotting.plot_times(
        models_times_and_voltages, Crate=1, linestyles=["k-", "b-.", "r--"]
    )
    file_name = "capacitance_solver_times.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight")


def discharge_times_and_errors(compute):
    savefile = "capacitance_times_and_errors.pickle"
    if compute:
        try:
            with open(savefile, "rb") as f:
                models_times_and_voltages = pickle.load(f)
        except FileNotFoundError:
            models_times_and_voltages = pybamm.get_infinite_nested_dict()
        models = [
            pybamm.lead_acid.NewmanTiedemann(name="direct form"),
            pybamm.lead_acid.NewmanTiedemann(
                {"surface form": "differential"},
                name="capacitance form\n(differential)",
            ),
            pybamm.lead_acid.NewmanTiedemann(
                {"surface form": "algebraic"}, name="capacitance form\n(algebraic)"
            ),
        ]
        Crates = [1]
        all_npts = np.linspace(10, 100, 2)
        t_eval = np.linspace(0, 1, 100)
        new_models_times_and_voltages = convergence_study(
            models, Crates, all_npts, t_eval
        )
        models_times_and_voltages.update(new_models_times_and_voltages)
        with open(savefile, "wb") as f:
            pickle.dump(models_times_and_voltages, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(savefile, "rb") as f:
                models_times_and_voltages = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute' first to generate results"
            )
    # plot_errors(models_times_and_voltages)
    plot_times(models_times_and_voltages)


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    # discharge_states(args.compute)
    discharge_times_and_errors(args.compute)
    plt.show()
