#
# Simulations: discharge of a lead-acid battery
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
import shared_plotting_2D
from collections import defaultdict
from shared_solutions_2D import model_comparison, convergence_study

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = None


def plot_errors(models_times_and_voltages):
    npts = 20
    linestyles = ["k-", "g--", "r:", "b-."]
    Crates = defaultdict(list)
    voltage_errors = defaultdict(list)
    fig, ax = plt.subplots(1, 1)
    for i, (model, times_and_voltages) in enumerate(models_times_and_voltages.items()):
        if model != "Full":
            for Crate, variables in times_and_voltages[npts].items():
                Crates[model].append(Crate)
                full_voltage = models_times_and_voltages["Full"][npts][Crate][
                    "Battery voltage [V]"
                ]
                reduced_voltage = variables["Battery voltage [V]"]
                voltage_errors[model].append(pybamm.rmse(full_voltage, reduced_voltage))
            ax.semilogx(
                Crates[model], voltage_errors[model], linestyles[i], label=model
            )
    ax.set_xlabel("C-rate")
    ax.set_ylabel("RMSE [V]")
    ax.legend(loc="best")
    fig.tight_layout()
    file_name = "discharge_asymptotics_rmse.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_times(models_times_and_voltages):
    shared_plotting_2D.plot_times(models_times_and_voltages, Crate=1)
    file_name = "discharge_asymptotics_solver_times.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def discharge_errors(compute):
    savefile = "2d_discharge_asymptotics_errors.pickle"
    if compute:
        try:
            with open(savefile, "rb") as f:
                models_times_and_voltages = pickle.load(f)
        except FileNotFoundError:
            models_times_and_voltages = pybamm.get_infinite_nested_dict()
        models = [
            pybamm.lead_acid.NewmanTiedemann(name="1D Full"),
            pybamm.lead_acid.LOQS(name="1D LOQS"),
            pybamm.lead_acid.Composite(name="1D Composite"),
            pybamm.lead_acid.Composite(
                {"current collector": "potential pair quite conductive averaged"},
                name="1D Composite",
            ),
            pybamm.lead_acid.NewmanTiedemann(
                {"dimensionality": 1, "current collector": "potential pair"},
                name="1D Full",
            ),
            pybamm.lead_acid.LOQS(
                {"dimensionality": 1, "current collector": "potential pair"},
                name="1D LOQS",
            ),
            pybamm.lead_acid.Composite(
                {"dimensionality": 1, "current collector": "potential pair"},
                name="1D Composite",
            ),
        ]
        Crates = np.logspace(0.01, 5, 5)
        sigmas = np.logspace(8000, 100 * 8000, 5)
        t_eval = np.linspace(0, 1, 100)
        new_models_times_and_voltages = convergence_study(
            models, Crates, sigmas, t_eval
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
    plot_errors(models_times_and_voltages)


if __name__ == "__main__":
    pybamm.set_logging_level("DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    discharge_states(args.compute)
    # discharge_times_and_errors(args.compute)
    plt.show()
