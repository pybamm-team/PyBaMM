#
# Times and errors discharge of a lead-acid battery
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
from shared_solutions import error_comparison, time_comparison

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = None


def plot_errors_and_times(model_voltages, model_times):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3))

    # Times
    linestyles = {"Full": "k-", "LOQS": "g--", "FOQS": "b:", "Composite": "r-."}
    plots = [None] * len(linestyles)
    models = list(linestyles.keys())
    # ntps is number of points in each electrode
    all_npts = [x * 3 for x in model_times[models[0]].keys()]
    for i, model in enumerate(models):
        times = list(model_times[model].values())
        plots[i], = ax2.loglog(all_npts, times, linestyles[model], label=model)
        ax2.set_xlim(min(all_npts), max(all_npts))
        ax2.set_xlabel("Number of grid points")
        ax2.set_ylabel("Solver time [s]")
        ax2.set_title("\\textbf{(b)} Times")

    # Errors
    Crates = list(model_voltages[models[0]].keys())
    errors = np.zeros(len(Crates))
    for i, model in enumerate(models):
        if model != "Full":
            Crates_variables = model_voltages[model]
            for j, (Crate, reduced_voltage) in enumerate(Crates_variables.items()):
                full_voltage = model_voltages["Full"][Crate]
                errors[j] = pybamm.rmse(full_voltage, reduced_voltage)
            ax1.loglog(Crates, errors, linestyles[model])
    ax1.set_xlim(min(Crates), max(Crates))
    ax1.set_xlabel("C-rate")
    ax1.set_ylabel("RMSE [V]")
    ax1.set_label("log(RMSE) [V]")
    ax1.set_title("\\textbf{(a)} Errors")

    leg = fig.legend(plots, models, loc="lower center", ncol=len(models))
    plt.subplots_adjust(bottom=0.3, right=0.95, wspace=0.5)
    leg.get_frame().set_edgecolor("k")

    # Save
    file_name = "1d_times_errors.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def discharge_times_and_errors(compute_times, compute_errors):
    savefile_errors = "1d_discharge_asymptotics_errors.pickle"
    savefile_times = "1d_discharge_asymptotics_times.pickle"
    models = [
        pybamm.lead_acid.NewmanTiedemann({"surface form": "algebraic"}, name="Full"),
        pybamm.lead_acid.LOQS(name="LOQS"),
        pybamm.lead_acid.FOQS(name="FOQS"),
        pybamm.lead_acid.CompositeExtended(name="Composite"),
    ]
    if compute_errors:
        Crates = np.logspace(np.log10(0.01), np.log10(10), 10)
        t_eval = np.linspace(0, 1, 100)
        model_voltages = error_comparison(models, Crates, t_eval)
        with open(savefile_errors, "wb") as f:
            pickle.dump(model_voltages, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(savefile_errors, "rb") as f:
                model_voltages = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute-errors' first to generate results"
            )
    if compute_times:
        all_npts = np.logspace(0.5, np.log10(500), 10)
        t_eval = np.linspace(0, 0.6, 100)
        Crate = 1
        model_times = time_comparison(models, Crate, all_npts, t_eval)
        with open(savefile_times, "wb") as f:
            pickle.dump(model_times, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(savefile_times, "rb") as f:
                model_times = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute-times' first to generate results"
            )
    plot_errors_and_times(model_voltages, model_times)


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compute-errors", action="store_true", help="(Re)-compute error results."
    )
    parser.add_argument(
        "--compute-times", action="store_true", help="(Re)-compute time results."
    )
    args = parser.parse_args()
    discharge_times_and_errors(args.compute_times, args.compute_errors)
    plt.show()
