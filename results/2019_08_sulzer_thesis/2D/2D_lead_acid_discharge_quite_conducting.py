#
# Simulations: discharge of a lead-acid battery
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
import shared_plotting_2D
from shared_solutions_2D import model_comparison

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = None


def plot_voltages(all_variables, t_eval):
    Crates = [0.1, 1, 2]
    sigmas = [5 * 8000, 10 * 8000, 100 * 8000]
    all_variables = {
        k: {sigma: models for sigma, models in v.items() if sigma in sigmas}
        for k, v in all_variables.items()
        if k in Crates
    }
    linestyles = ["k:", "k-", "g--", "b-."]
    linewidths = [0.7, 1.4, 1.4, 1.4]
    shared_plotting_2D.plot_voltages(
        all_variables, t_eval, linestyles=linestyles, linewidths=linewidths
    )
    file_name = "2d_quite_discharge_voltage_comparison.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_voltage_components(all_variables, t_eval):
    Crate = 1
    sigmas = [5 * 8000, 10 * 8000, 100 * 8000]
    model = "1+1D Composite"
    all_variables = {
        sigma: models
        for sigma, models in all_variables[Crate].items()
        if sigma in sigmas
    }
    shared_plotting_2D.plot_voltage_components(all_variables, t_eval, model, sigmas)
    file_name = "2d_discharge_voltage_components.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def discharge_states(compute):
    savefile = "2d_quite_discharge_asymptotics_data.pickle"
    models = [
        pybamm.lead_acid.NewmanTiedemann(
            {"surface form": "algebraic", "dimensionality": 1}, name="1D Full"
        ),
        pybamm.lead_acid.NewmanTiedemann(
            {"dimensionality": 1, "current collector": "potential pair"},
            name="1+1D Full",
        ),
        pybamm.lead_acid.LOQS(
            {
                "dimensionality": 1,
                "current collector": "potential pair quite conductive",
            },
            name="1+1D LOQS",
        ),
        # pybamm.lead_acid.FOQS(
        #     {"dimensionality": 1, "current collector": "potential pair"},
        #     name="FOQS",
        # ),
        pybamm.lead_acid.CompositeExtended(
            {
                "dimensionality": 1,
                "current collector": "potential pair quite conductive averaged",
            },
            name="1+1D Composite",
        ),
    ]
    variables_to_keep = [
        "x",
        "x [m]",
        "z",
        "z [m]",
        "Time",
        "Time [h]",
        "Average battery reaction overpotential [V]",
        "Average battery concentration overpotential [V]",
        "Average battery electrolyte ohmic losses [V]",
        "Battery current collector overpotential [V]",
        "Battery voltage [V]",
        "Electrolyte concentration [Molar]",
        "X-averaged electrolyte concentration [Molar]",
        "Oxygen concentration [Molar]",
        "X-averaged oxygen concentration [Molar]",
        "Electrolyte potential [V]",
        "X-averaged electrolyte potential [V]",
        "Current collector current density",
        "State of Charge",
        "Fractional Charge Input",
    ]
    for model in models:
        model.variables = {
            name: variable
            for name, variable in model.variables.items()
            if name in variables_to_keep
        }
    Crates = [0.1, 1, 2]
    sigmas = [8000, 5 * 8000, 10 * 8000, 100 * 8000]

    t_eval = np.linspace(0, 1, 100)
    extra_parameter_values = {}
    all_variables, t_eval = model_comparison(
        models,
        Crates,
        sigmas,
        t_eval,
        savefile,
        use_force=compute,
        extra_parameter_values=extra_parameter_values,
    )
    plot_voltages(all_variables, t_eval)
    # plot_voltage_components(all_variables, t_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    if args.compute:
        pybamm.set_logging_level("DEBUG")
    discharge_states(args.compute)
    # discharge_times_and_errors(args.compute)
    plt.show()
