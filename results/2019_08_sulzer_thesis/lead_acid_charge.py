#
# Simulations: charge of a lead-acid battery
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
    shared_plotting.plot_voltages(all_variables, t_eval)
    file_name = "charge_voltage_comparison.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_interfacial_currents(models_variables, t_eval):
    models = ["Full", "LOQS"]
    file_name = "charge_interfacial_current_density_comparison.eps"
    fig, ax = plt.subplots(1, 1)
    output_vars = [
        "Average positive electrode interfacial current density",
        "Average positive electrode oxygen interfacial current density",
        "Average negative electrode oxygen interfacial current density",
        "Average negative electrode interfacial current density",
    ]
    labels = [
        "Positive electrode (main)",
        "Positive electrode (oxygen)",
        "Negative electrode (oxygen)",
        "Negative electrode (main)",
    ]
    t_max = max(np.nanmax(var["Time [h]"](t_eval)) for var in models_variables.values())
    ax.set_xlim([0, t_max])
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Interfacial current densities")
    linestyles = ["--", ":", "-.", "-"]
    colors = ["k", "r"]
    plots = {}
    for j, (model, variables) in enumerate(models_variables.items()):
        if model in models:
            for k, var in enumerate(output_vars):
                plots[(model, k)], = ax.plot(
                    variables["Time [h]"](t_eval),
                    variables[var](t_eval),
                    linestyle=linestyles[k],
                    color=colors[j],
                )
    leg1 = ax.legend(
        [plots[("Full", 3)], plots[("LOQS", 3)]],
        ["Full", "LOQS"],
        loc="center left",
        bbox_to_anchor=(1, 0.25),
    )
    ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.75))
    ax.add_artist(leg1)
    fig.tight_layout()
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
        shared_plotting.plot_variable(all_variables, times, var, exceptions)
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
            pybamm.lead_acid.FOQS(
                {"surface form": "algebraic", "side reactions": ["oxygen"]}, name="FOQS"
            ),
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
    # plot_interfacial_currents(all_variables[-1], t_eval)
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
