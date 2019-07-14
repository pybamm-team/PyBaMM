#
# Simulations: charge of a lead-acid battery
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
from config import OUTPUT_DIR
from shared import model_comparison, simulation


def plot_voltages(all_variables, t_eval, models, linestyles, file_name):
    # Plot
    n = int(len(all_variables) // np.sqrt(len(all_variables)))
    m = int(np.ceil(len(all_variables) / n))
    fig, axes = plt.subplots(n, m, figsize=(8, 4.5))
    labels = [model for model in [x for x in all_variables.values()][0].keys()]
    for k, (Crate, models_variables) in enumerate(all_variables.items()):
        ax = axes.flat[k]
        t_max = max(
            np.nanmax(var["Time [h]"](t_eval)) for var in models_variables.values()
        )
        ax.set_xlim([0, t_max])
        ax.set_ylim([12, 15])
        ax.set_xlabel("Time [h]")
        ax.set_title(
            "\\textbf{{{}C}} ($\\mathcal{{C}}_e={}$)".format(-Crate, -Crate * 0.6)
        )
        # ax.set_title(
        #     "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
        #         chr(97 + k), Crate, Crate * 0.6
        #     )
        # )

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        if k % m == 0:
            ax.set_ylabel("Voltage [V]")
        for j, (model, variables) in enumerate(models_variables.items()):
            if model in models:
                ax.plot(
                    variables["Time [h]"](t_eval),
                    variables["Terminal voltage [V]"](t_eval),
                    linestyles[j],
                )
    ax.legend(labels, bbox_to_anchor=(1.05, 2), loc=2)
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_interfacial_currents(models_variables, t_eval, models, file_name):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    output_vars = [
        "Average negative electrode interfacial current density",
        "Average positive electrode interfacial current density",
        "Average negative electrode oxygen interfacial current density",
        "Average positive electrode oxygen interfacial current density",
    ]
    labels = [
        "Negative current (main)",
        "Positive current (main)",
        "Negative current (oxygen)",
        "Positive current (oxygen)",
    ]
    t_max = max(np.nanmax(var["Time [h]"](t_eval)) for var in models_variables.values())
    ax.set_xlim([0, t_max])
    # ax.set_ylim([12, 15])
    ax.set_xlabel("Time [h]")
    ax.set_title("Interfacial current densities")
    linestyles = ["-", "--", "-.", ":"]
    colors = ["k", "r"]
    for j, (model, variables) in enumerate(models_variables.items()):
        if model in models:
            for k, var in enumerate(output_vars):
                ax.plot(
                    variables["Time [h]"](t_eval),
                    variables[var](t_eval),
                    linestyle=linestyles[k],
                    color=colors[j],
                )
    ax.legend(labels, loc=2, bbox_to_anchor=(1, 0.7))
    fig.tight_layout()
    # plt.show()
    plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def compare_voltages(compute):
    if compute:
        models = [
            pybamm.lead_acid.NewmanTiedemann(
                {"side reactions": ["oxygen"]}, name="Full"
            ),
            pybamm.lead_acid.LOQS(
                {"surface form": "algebraic", "side reactions": ["oxygen"]},
                name="Leading-order",
            ),
        ]
        Crates = [-0.1, -0.2, -0.5, -1, -2, -5]
        extra_parameter_values = {
            "Positive electrode"
            + "reference exchange-current density (oxygen) [A.m-2]": 1e-24,
            "Initial State of Charge": 0.5,
        }
        t_eval = np.linspace(0, 2, 100)
        all_variables, t_eval = model_comparison(
            models, Crates, t_eval, extra_parameter_values=extra_parameter_values
        )
        with open("charge_asymptotics_data.pickle", "wb") as f:
            data = (all_variables, t_eval)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open("charge_asymptotics_data.pickle", "rb") as f:
            (all_variables, t_eval) = pickle.load(f)
    import ipdb

    ipdb.set_trace()
    models = ["Full", "Leading-order"]
    linestyles = ["k-", "r--"]
    file_name = "charge_voltage_comparison.eps"
    # plot_voltages(all_variables, t_eval, models, linestyles, file_name)
    ###
    file_name = "charge_interfacial_currents.eps"
    models = ["Full"]
    plot_interfacial_currents(all_variables[-1], t_eval, models, file_name)
    file_name = "charge_interfacial_currents_comparison.eps"
    models = ["Full", "Leading-order"]
    plot_interfacial_currents(all_variables[-1], t_eval, models, file_name)


def plot_variables(compute, Crate, models, file_save, file_plot):
    t_eval = np.linspace(0, 1, 100)
    param = {"Typical current [A]": Crate * 17}
    if compute:
        models, mesh, solutions = simulation(
            models, t_eval, extra_parameter_values=param
        )
        with open(file_save, "wb") as f:
            pickle.dump(solutions, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(file_save, "rb") as f:
            model, mesh = simulation(
                models, t_eval, extra_parameter_values=param, disc_only=True
            )
            solutions = pickle.load(f)
    output_variables = [
        "Electrolyte concentration [mol.m-3]",
        "Oxygen concentration [mol.m-3]",
        "Interfacial current density [A.m-2]",
        "Terminal voltage [V]",
    ]
    plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)

    for t in range(0, 10):
        tt = t / 10
        plot.plot(tt, dynamic=False, figsize=(12, 7))
        # plt.show()
        plt.savefig(
            OUTPUT_DIR + "{}_t=0pt{}.eps".format(file_plot, t), format="eps", dpi=1000
        )


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()

    compare_voltages(args.compute)
    # Crates = [-1]
    # for Crate in Crates:
    #     models = [
    #         pybamm.lead_acid.NewmanTiedemann(
    #             {"side reactions": ["oxygen"]}, name="Full"
    #         ),
    #         pybamm.lead_acid.LOQS(
    #             {"surface form": "algebraic", "side reactions": ["oxygen"]},
    #             name="Leading-order",
    #         ),
    #     ]
    #     file_save = "charge_asymptotics_data_{}C.pickle".format(Crate)
    #     file_plot = "charge_states/leading_full_{}C".format(Crate)
    #     plot_variables(args.compute, Crate, models, file_save, file_plot)
    #     models = [
    #         pybamm.lead_acid.NewmanTiedemann(name="Without oxygen"),
    #         pybamm.lead_acid.NewmanTiedemann(
    #             {"side reactions": ["oxygen"]}, name="With oxygen"
    #         ),
    #     ]
    #     file_save = "charge_effect_of_side_reactions_data_{}C.pickle".format(Crate)
    #     file_plot = "charge_states/with_without_reactions_{}C".format(Crate)
    #     plot_variables(args.compute, Crate, models, file_save, file_plot)
