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


def plot_voltages(all_variables, t_eval):
    Crates = [0.1, 1, 2]
    all_variables = {k: v for k, v in all_variables.items() if k in Crates}
    linestyles = ["k:", "k-", "g--", "b-."]
    linewidths = [0.7, 1.4, 1.4, 1.4]
    shared_plotting_2D.plot_voltages(
        all_variables, t_eval, linestyles=linestyles, linewidths=linewidths
    )
    file_name = "2d_poor_discharge_voltage_comparison.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_variables(all_variables, t_eval):
    # Set up
    Crates = [0.1, 1, 2]
    times = np.array([0, 0.195, 0.375, 0.545])
    var_file_names = {
        "X-averaged electrolyte concentration [Molar]"
        + "": "2d_poor_discharge_average_electrolyte_concentration_comparison.eps",
        # "X-averaged electrolyte potential [V]"
        # + "": "2d_poor_discharge_average_electrolyte_potential_comparison.eps",
        "Current collector current density"
        + "": "2d_poor_current_collector_current_density_comparison.eps",
    }
    limits_exceptions = {"X-averaged electrolyte concentration [Molar]": {"min": 0}}
    linestyles = ["k:", "k-", "g--", "b-."]
    linewidths = [0.7, 1.4, 1.4, 1.4]
    all_variables = {k: v for k, v in all_variables.items() if k in Crates}
    for var, file_name in var_file_names.items():
        if var in limits_exceptions:
            exceptions = limits_exceptions[var]
        else:
            exceptions = {}
        shared_plotting_2D.plot_variable(
            all_variables,
            times,
            var,
            exceptions,
            linestyles=linestyles,
            linewidths=linewidths,
        )
        if OUTPUT_DIR is not None:
            plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_variables_x_z(all_variables, t_eval):
    # Set up
    Crates = [0.1, 1, 2]
    times = np.array([0, 0.195, 0.375, 0.545])
    var_file_names = {
        "Electrolyte concentration [Molar]"
        + "": "2d_poor_discharge_electrolyte_concentration_comparison.eps"
    }
    limits_exceptions = {"Electrolyte concentration [Molar]": {"min": 0}}
    all_variables = {k: v for k, v in all_variables.items() if k in Crates}
    for var, file_name in var_file_names.items():
        if var in limits_exceptions:
            exceptions = limits_exceptions[var]
        else:
            exceptions = {}
        shared_plotting_2D.plot_variable_x_z(all_variables, times, var, exceptions)
        if OUTPUT_DIR is not None:
            plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def discharge_states(compute):
    savefile = "2d_poor_discharge_asymptotics_data.pickle"
    if compute:
        models = [
            pybamm.lead_acid.NewmanTiedemann(
                {"surface form": "differential", "dimensionality": 1}, name="1D Full"
            ),
            pybamm.lead_acid.NewmanTiedemann(
                {"dimensionality": 1, "current collector": "potential pair"},
                name="1+1D Full",
            ),
            pybamm.lead_acid.LOQS(
                {"dimensionality": 1, "current collector": "potential pair"},
                name="1+1D LOQS",
            ),
            # pybamm.lead_acid.FOQS(
            #     {"dimensionality": 1, "current collector": "potential pair"},
            #     name="FOQS",
            # ),
            pybamm.lead_acid.CompositeExtended(
                {"dimensionality": 1, "current collector": "potential pair"},
                name="1+1D Composite",
            ),
        ]
        Crates = [0.1, 1, 2]
        sigmas = [10 * 8000]  # , 100 * 8000, 1000 * 8000]

        t_eval = np.linspace(0, 1, 100)
        extra_parameter_values = {}  # "Bruggeman coefficient": 0.001}
        all_variables, t_eval = model_comparison(
            models,
            Crates,
            sigmas,
            t_eval,
            extra_parameter_values=extra_parameter_values,
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
    # plot_voltages(all_variables, t_eval)
    # plot_variables(all_variables, t_eval)
    plot_variables_x_z(all_variables, t_eval)


if __name__ == "__main__":
    pybamm.set_logging_level("DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    discharge_states(args.compute)
    plt.show()
