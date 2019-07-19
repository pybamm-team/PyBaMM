#
# Simulations: discharge of a lead-acid battery
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
import shared_plotting
from collections import defaultdict
from shared_solutions import model_comparison, convergence_study

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = None


def nested_dict():
    return defaultdict(nested_dict)


def plot_voltages(all_variables, t_eval):
    shared_plotting.plot_voltages(all_variables, t_eval)
    file_name = "discharge_voltage_comparison.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight")


def plot_variables(all_variables, t_eval):
    # Set up
    Crates = [0.1, 2, 5]
    times = np.linspace(0, 0.5, 4)
    var_file_names = {
        "Electrolyte concentration [Molar]": "discharge_electrolyte_concentration_comparison.eps",
        "Electrolyte potential [V]": "discharge_electrolyte_potential_comparison.eps",
        "Interfacial current density": "discharge_interfacial_current_density_comparison.eps",
    }
    limits_exceptions = {"Electrolyte concentration [Molar]": {"min": 0}}
    for var, file_name in var_file_names.items():
        if var in limits_exceptions:
            exceptions = limits_exceptions[var]
        else:
            exceptions = {}
        shared_plotting.plot_variable(
            all_variables, t_eval, times, Crates, var, exceptions
        )
        if OUTPUT_DIR is not None:
            plt.savefig(
                OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight"
            )


def plot_voltage_breakdown(all_variables, t_eval):
    Crates = [0.1, 2, 5]
    model = "Composite"
    shared_plotting.plot_voltage_breakdown(all_variables, t_eval, model, Crates)
    file_name = "discharge_voltage_breakdown.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight")


def lead_acid_discharge_states(compute):
    if compute:
        models = [
            pybamm.lead_acid.NewmanTiedemann(name="Full"),
            pybamm.lead_acid.LOQS(name="LOQS"),
            pybamm.lead_acid.FOQS(name="FOQS"),
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
    else:
        try:
            with open("discharge_asymptotics_data.pickle", "rb") as f:
                (all_variables, t_eval) = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute' first to generate results"
            )
    plot_voltages(all_variables, t_eval)
    plot_variables(all_variables, t_eval)
    plot_voltage_breakdown(all_variables, t_eval)


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
                    "Terminal voltage [V]"
                ]
                reduced_voltage = variables["Terminal voltage [V]"]
                voltage_errors[model].append(pybamm.rmse(full_voltage, reduced_voltage))
            ax.semilogx(
                Crates[model], voltage_errors[model], linestyles[i], label=model
            )
    ax.set_xlabel("C-rate")
    ax.set_ylabel("RMSE [V]")
    ax.legend(loc="best")
    file_name = "discharge_asymptotics_rmse.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight")


def plot_times(models_times_and_voltages):
    Crate = 1
    linestyles = ["k-", "g--", "r:", "b-."]
    all_npts = defaultdict(list)
    solver_time = defaultdict(list)
    fig, ax = plt.subplots(1, 1)
    for i, (model, times_and_voltages) in enumerate(models_times_and_voltages.items()):
        for npts, Crates_variables in times_and_voltages.items():
            all_npts[model].append(npts * 3)
            solver_time = times_and_voltages[npts][Crate]["solution object"].solve_time
            solver_time[model].append(solver_time)
        ax.loglog(all_npts[model], solver_time[model], linestyles[i], label=model)
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("Solver time [s]")
    ax.legend(loc="best")
    file_name = "discharge_asymptotics_solver_times.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000, bbox_inches="tight")


def lead_acid_discharge_times_and_errors(compute):
    savefile = "discharge_asymptotics_times_and_errors.pickle"
    if compute:
        try:
            with open(savefile, "rb") as f:
                models_times_and_voltages = pickle.load(f)
        except FileNotFoundError:
            models_times_and_voltages = nested_dict()
        models_times_and_voltages = nested_dict()
        models = [
            pybamm.lead_acid.NewmanTiedemann(
                {"surface form": "algebraic"}, name="Full"
            ),
            pybamm.lead_acid.LOQS(name="LOQS"),
            # pybamm.lead_acid.FOQS(name="FOQS"),
            # pybamm.lead_acid.Composite(name="Composite"),
        ]
        Crates = np.linspace(0.01, 5, 2)
        all_npts = [20]
        t_eval = np.linspace(0, 1, 100)
        new_models_times_and_voltages = convergence_study(
            models, Crates, all_npts, t_eval
        )
        import ipdb

        ipdb.set_trace()
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
    # plot_times(models_times_and_voltages)


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    # lead_acid_discharge_states(args.compute)
    lead_acid_discharge_times_and_errors(args.compute)
    plt.show()
