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
from shared_solutions_2D import model_comparison, variables_to_keep

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = None


def plot_voltages(all_variables, t_eval):
    Crates = [0.1, 1, 2]
    sigmas = [8000, 5 * 8000, 10 * 8000, 100 * 8000]
    all_variables = {
        k: {sigma: models for sigma, models in v.items() if sigma in sigmas}
        for k, v in all_variables.items()
        if k in Crates
    }
    linestyles = ["k:", "k-", "g--", "r-."]
    linewidths = [0.7, 1.4, 1.4, 1.4]
    shared_plotting_2D.plot_voltages(
        all_variables,
        t_eval,
        linestyles=linestyles,
        linewidths=linewidths,
        figsize=(6.4, 5),
    )
    file_name = "2d_poor_discharge_voltage_comparison.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_variables(all_variables, t_eval):
    # Set up
    Crates = [0.1, 1, 2]
    times = np.array([0, 0.078, 0.156])
    var_file_names = {
        "X-averaged electrolyte concentration [Molar]"
        + "": "2d_poor_discharge_average_electrolyte_concentration_comparison"
    }
    limits_exceptions = {"X-averaged electrolyte concentration [Molar]": {"min": 0}}
    linestyles = ["k:", "k-", "g--", "r-."]
    linewidths = [0.7, 1.4, 1.4, 1.4]
    all_variables = {k: v for k, v in all_variables.items() if k in Crates}
    for var, file_name in var_file_names.items():
        if var in limits_exceptions:
            exceptions = limits_exceptions[var]
        else:
            exceptions = {}
        # for sigma in [8000, 5 * 8000, 10 * 8000]:
        #     shared_plotting_2D.plot_variable(
        #         all_variables,
        #         times,
        #         sigma,
        #         var,
        #         exceptions,
        #         linestyles=linestyles,
        #         linewidths=linewidths,
        #     )
        #     if OUTPUT_DIR is not None:
        #         plt.savefig(
        #             OUTPUT_DIR + file_name + "_sigma={}.eps".format(sigma),
        #             format="eps",
        #             dpi=1000,
        #         )
        shared_plotting_2D.plot_variable_allsigma(
            all_variables,
            0.1,
            var,
            exceptions,
            linestyles=linestyles,
            linewidths=linewidths,
            figsize=(6.5, 5),
        )
        if OUTPUT_DIR is not None:
            plt.savefig(
                OUTPUT_DIR + file_name + "_allsigma.eps", format="eps", dpi=1000
            )


def plot_variables_x_z(all_variables, t_eval):
    # Set up
    var_file_names = {
        "Electrolyte concentration [Molar]"
        + "": "2d_poor_discharge_electrolyte_concentration_comparison"
    }
    limits_exceptions = {"Electrolyte concentration [Molar]": {"min": 0}}
    for var, file_name in var_file_names.items():
        if var in limits_exceptions:
            exceptions = limits_exceptions[var]
        else:
            exceptions = {}
        for sigma in [8000, 5 * 8000, 10 * 8000]:
            time_Crate_sigma = (0.1, 1, sigma)
            shared_plotting_2D.plot_variable_x_z(
                all_variables, time_Crate_sigma, var, exceptions
            )
            if OUTPUT_DIR is not None:
                plt.savefig(
                    OUTPUT_DIR + file_name + "_sigma={}.eps".format(sigma),
                    format="eps",
                    dpi=1000,
                )


def discharge_states(compute):
    savefile = "2d_poor_discharge_asymptotics_data.pickle"
    models = [
        pybamm.lead_acid.NewmanTiedemann(
            {"surface form": "algebraic", "dimensionality": 1}, name="1D Full"
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
    for model in models:
        model.variables = {
            name: variable
            for name, variable in model.variables.items()
            if name in variables_to_keep
        }
        model.events = {
            name: event
            for name, event in model.events.items()
            if name != "Minimum voltage"
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
    plot_variables(all_variables, t_eval)
    plot_variables_x_z(all_variables, t_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    pybamm.set_logging_level("DEBUG")
    discharge_states(args.compute)
    plt.show()
