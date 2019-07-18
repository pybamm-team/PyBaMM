#
# Simulations: discharge of a lead-acid battery
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
    # Plot
    linestyles = ["k-", "g--", "r:", "b-."]
    file_name = "discharge_voltage_comparison.eps"
    n = int(len(all_variables) // np.sqrt(len(all_variables)))
    m = int(np.ceil(len(all_variables) / n))
    fig, axes = plt.subplots(n, m, figsize=(6.4, 4.5))
    labels = [model for model in [x for x in all_variables.values()][0].keys()]
    y_min = min(
        np.nanmin(variables["Terminal voltage [V]"](t_eval))
        for models_variables in all_variables.values()
        for variables in models_variables.values()
    )
    y_max = 1.02 * max(
        np.nanmax(variables["Terminal voltage [V]"](t_eval))
        for models_variables in all_variables.values()
        for variables in models_variables.values()
    )
    for k, (Crate, models_variables) in enumerate(all_variables.items()):
        ax = axes.flat[k]
        t_max = max(
            np.nanmax(var["Time [h]"](t_eval)) for var in models_variables.values()
        )
        ax.set_xlim([0, t_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Time [h]")
        ax.set_title(
            "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
                chr(97 + k), Crate, Crate * 0.6
            )
        )

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        if k % m == 0:
            ax.set_ylabel("Voltage [V]")
        for j, (model, variables) in enumerate(models_variables.items()):
            ax.plot(
                variables["Time [h]"](t_eval),
                variables["Terminal voltage [V]"](t_eval),
                linestyles[j],
            )
    leg = fig.legend(labels, loc="lower center", ncol=len(labels), frameon=True)
    leg.get_frame().set_edgecolor("k")
    plt.subplots_adjust(bottom=0.25, right=0.95, hspace=1.1, wspace=0.4)
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
            all_variables, t_eval, times, Crates, var, file_name, exceptions
        )


def plot_voltage_breakdown(all_variables, t_eval):
    # Plot
    Crates = [0.1, 2, 5]
    model = "Composite"
    linestyles = ["k-", "g--", "r:", "b-."]
    file_name = "discharge_voltage_breakdown.eps"
    n = int(len(Crates) // np.sqrt(len(Crates)))
    m = int(np.ceil(len(Crates) / n))
    fig, axes = plt.subplots(n, m, figsize=(6.4, 2.3))
    labels = ["V", "$V_U$", "$V_k$", "$V_c$", "$V_o$"]
    overpotentials = [
        "Average reaction overpotential [V]",
        "Average concentration overpotential [V]",
        "Average electrolyte ohmic losses [V]",
    ]
    y_min = 0.95 * min(
        np.nanmin(models_variables[model]["Terminal voltage [V]"](t_eval))
        for models_variables in all_variables.values()
    )
    y_max = 1.05 * max(
        np.nanmax(models_variables[model]["Terminal voltage [V]"](t_eval))
        for models_variables in all_variables.values()
    )
    for k, Crate in enumerate(Crates):
        variables = all_variables[Crate][model]
        ax = axes.flat[k]

        # Set up
        t_max = np.nanmax(variables["Time [h]"](t_eval))
        ax.set_xlim([0, t_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Time [h]")
        ax.set_title(
            "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
                chr(97 + k), Crate, Crate * 0.6
            )
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        if k % m == 0:
            ax.set_ylabel("Voltage [V]")

        # Plot
        # Initialise
        time = variables["Time [h]"](t_eval)
        initial_ocv = variables["Average open circuit voltage [V]"](0) * 6
        ocv = variables["Average open circuit voltage [V]"](t_eval) * 6
        ax.fill_between(time, ocv, initial_ocv)
        top = ocv
        # Plot
        for j, overpotential in enumerate(overpotentials):
            bottom = top + variables[overpotential](t_eval) * 6
            ax.fill_between(time, bottom, top)
            top = bottom
        ax.plot(time, variables["Terminal voltage [V]"](t_eval), "k--")
    leg = axes.flat[-1].legend(
        labels, bbox_to_anchor=(1.05, 0.5), loc="center left", frameon=True
    )
    leg.get_frame().set_edgecolor("k")
    fig.tight_layout()
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


def lead_acid_discharge_times_and_errors(compute):
    if compute:
        models = [
            pybamm.lead_acid.NewmanTiedemann(name="Full"),
            pybamm.lead_acid.LOQS(name="LOQS"),
            pybamm.lead_acid.FOQS(name="FOQS"),
            pybamm.lead_acid.Composite(name="Composite"),
        ]
        Crates = [0.1, 0.2]
        npts = [10, 20]
        t_eval = np.linspace(0, 1, 100)
        times_and_voltages = model_comparison(models, Crates, t_eval)
        with open("discharge_asymptotics_times_and_errors.pickle", "wb") as f:
            data = times_and_voltages
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open("discharge_asymptotics_times_and_errors.pickle", "rb") as f:
                times_and_voltages = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute' first to generate results"
            )
    plot_errors(times_and_voltages)


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    # lead_acid_discharge_states(args.compute)
    lead_acid_discharge_times_and_errors(args.compute)
    plt.show()
