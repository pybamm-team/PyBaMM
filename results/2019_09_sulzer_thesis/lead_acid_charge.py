#
# Charge of a lead-acid battery
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
    Crates = [-0.1, -0.2, -0.5, -1, -2, -5]
    all_variables = {k: v for k, v in all_variables.items() if k in Crates}
    shared_plotting.plot_voltages(
        all_variables, t_eval, linestyles=["k-", "g--", "r-."]
    )
    file_name = "charge_voltage_comparison.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_interfacial_currents(all_variables, t_eval):
    Crates = [-0.1, -2, -5]
    all_variables = {Crate: v for Crate, v in all_variables.items() if Crate in Crates}
    file_name = "charge_interfacial_current_density_comparison.eps"
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
        all_variables,
        t_eval,
        output_vars,
        labels,
        colors=["k", "g", "r"],
        figsize=(6.4, 4),
    )
    plt.subplots_adjust(
        bottom=0.28, left=0.2, right=0.95, wspace=0.3, hspace=0.4, top=0.9
    )
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_variables(all_variables, t_eval):
    # Set up
    Crates = [-0.1, -2, -5]
    times = np.linspace(0, 2, 4)
    var_file_names = {
        "Electrolyte concentration [Molar]"
        + "": "charge_electrolyte_concentration_comparison.eps",
        "Oxygen concentration [Molar]": "charge_oxygen_concentration_comparison.eps",
    }
    limits_exceptions = {"Electrolyte concentration [Molar]": {"min": 0}}
    all_variables = {k: v for k, v in all_variables.items() if k in Crates}
    for var, file_name in var_file_names.items():
        if var in limits_exceptions:
            exceptions = limits_exceptions[var]
        else:
            exceptions = {}
        shared_plotting.plot_variable(
            all_variables,
            times,
            var,
            exceptions,
            linestyles=["k-", "g--", "r-."],
            yaxis="FCI",
        )
        if OUTPUT_DIR is not None:
            plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_voltage_components(all_variables, t_eval):
    Crates = [-0.1, -2, -5]
    model = "Composite"
    shared_plotting.plot_voltage_components(all_variables, t_eval, model, Crates)
    file_name = "charge_voltage_components.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def charge_states(compute):
    if compute:
        models = [
            pybamm.lead_acid.NewmanTiedemann(
                {"side reactions": ["oxygen"]}, name="Full"
            ),
            pybamm.lead_acid.LOQS(
                {"surface form": "algebraic", "side reactions": ["oxygen"]}, name="LOQS"
            ),
            # pybamm.lead_acid.FOQS(
            #     {"surface form": "algebraic", "side reactions": ["oxygen"]}, name="FOQS"
            # ),
            # pybamm.lead_acid.Composite(
            #     {"surface form": "algebraic", "side reactions": ["oxygen"]},
            #     name="Composite",
            # ),
            pybamm.lead_acid.CompositeExtended(
                {"surface form": "algebraic", "side reactions": ["oxygen"]},
                name="Composite",
            ),
        ]
        Crates = [-0.1, -0.2, -0.5, -1, -2, -5]
        t_eval = np.linspace(0, 3, 100)
        extra_parameter_values = {
            "Positive electrode"
            + "reference exchange-current density (oxygen) [A.m-2]": 1e-24,
            "Initial State of Charge": 0.5,
        }
        all_variables, t_eval = model_comparison(
            models, Crates, t_eval, extra_parameter_values=extra_parameter_values
        )
        with open("charge_asymptotics_data.pickle", "wb") as f:
            data = (all_variables, t_eval)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open("charge_asymptotics_data.pickle", "rb") as f:
                (all_variables, t_eval) = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute' first to generate results"
            )
    plot_voltages(all_variables, t_eval)
    plot_interfacial_currents(all_variables, t_eval)
    plot_variables(all_variables, t_eval)
    plot_voltage_components(all_variables, t_eval)


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    charge_states(args.compute)
    # charge_times_and_errors(args.compute)
    plt.show()
