#
# Simulations: discharge of a lead-acid battery
#
import matplotlib.pyplot as plt
import numpy as np
import pybamm
from config import OUTPUT_DIR


def velocity_comparison(models, Crates):
    # load parameter values and geometry
    geometry = models[0].default_geometry
    param = models[0].default_parameter_values

    # Process parameters (same parameters for all models)
    for model in models:
        param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5}
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)

    # discretise models
    discs = {}
    for model in models:
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        # Store discretisation
        discs[model] = disc

    # solve model for range of Crates
    all_variables = {}
    for Crate in Crates:
        all_variables[Crate] = {}
        current = Crate * 17
        param.update({"Typical current [A]": current})
        t_eval = np.linspace(0, 1, 100)
        for model in models:
            param.update_model(model, discs[model])
            solver = model.default_solver
            solver.solve(model, t_eval)
            all_variables[Crate][model] = pybamm.post_process_variables(
                model.variables, solver.t, solver.y, mesh
            )

    return all_variables, t_eval


def plot_voltages(all_variables, t_eval):
    # Plot
    plt.subplots(figsize=(6, 4))
    n = int(len(all_variables) // np.sqrt(len(all_variables)))
    m = np.ceil(len(all_variables) / n)
    for k, Crate in enumerate(all_variables.keys()):
        models_variables = all_variables[Crate]
        t_max = max(
            np.nanmax(var["Time [h]"](t_eval)) for var in models_variables.values()
        )
        ax = plt.subplot(n, m, k + 1)
        plt.axis([0, t_max, 10.5, 13])
        plt.xlabel("Time [h]")
        plt.title("\\textbf{{({})}} {}C".format(chr(97 + k), Crate))

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        for model, variables in models_variables.items():
            if k == 0:
                label = model.name
            else:
                label = None
            if k % m == 0:
                plt.ylabel("Voltage [V]")

            plt.plot(
                variables["Time [h]"](t_eval),
                variables["Terminal voltage [V]"](t_eval) * 6,
                label=label,
            )
        # plt.legend(loc="upper right")
    # file_name = "convection_voltage_comparison.eps".format(Crate)
    # plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)
    plt.show()


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")
    models = [
        pybamm.lead_acid.NewmanTiedemann({"convection": True}),
        pybamm.lead_acid.NewmanTiedemann(),
    ]
    Crates = [0.1]  # 0.1, 0.2, 0.5, 1, 2, 5]
    all_variables, t_eval = velocity_comparison(models, Crates)
    plot_voltages(all_variables, t_eval)
