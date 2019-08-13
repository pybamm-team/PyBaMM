#
# Figure 2: DFN, SPMe, SPM voltage comparison
#
import pybamm
import numpy as np
import matplotlib.pyplot as plt

generate_plots = True
print_RMSE = True
export_data = True

#
# Load models set mesh etc
#

# load models
models = [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.SPMe(), pybamm.lithium_ion.DFN()]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
for model in models:
    param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}

# discretise models
discs = []
for model in models:
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    discs.append(disc)


#
# Loop over c-rates
#

C_rates = [0.1, 0.5, 1, 2, 3]

colour = {0.1: "r", 0.5: "b", 1: "g", 2: "m", 3: "y"}

variables = ["Discharge capacity [A.h.m-2]", "Terminal voltage [V]"]

if export_data:
    dir_path = "results/2019_08_asymptotic_spme/data/figure_2"
    exporter = pybamm.ExportCSV(dir_path)

RMSE = {"SPM-DFN": [], "SPMe-DFN": []}

for C_rate in C_rates:

    param["Typical current [A]"] = (
        C_rate * 24 * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
    )

    # solve model
    solutions = [None] * len(models)
    t_eval = np.linspace(0, 0.17, 100)

    for i, model in enumerate(models):
        param.update_model(model, discs[i])
        solutions[i] = model.default_solver.solve(model, t_eval)

    if generate_plots:

        for i, model in enumerate(models):
            t, y = solutions[i].t, solutions[i].y
            discharge_cap = pybamm.ProcessedVariable(
                model.variables[variables[0]], t, y, discs[i].mesh
            )(t)
            voltage = pybamm.ProcessedVariable(
                model.variables[variables[1]], t, y, discs[i].mesh
            )

            if i == 0:
                plt.plot(
                    discharge_cap, voltage(t), lw=2, linestyle=":", color=colour[C_rate]
                )
            elif i == 1:
                plt.plot(
                    discharge_cap,
                    voltage(t),
                    lw=2,
                    linestyle="--",
                    color=colour[C_rate],
                )
            elif i == 2:
                plt.plot(
                    discharge_cap, voltage(t), lw=2, label=C_rate, color=colour[C_rate]
                )

        plt.xlabel(variables[0], fontsize=15)
        plt.ylabel(variables[1], fontsize=15)
        plt.legend(fontsize=15)

    if print_RMSE:

        spm = models[0]
        t_spm = solutions[0].t
        y_spm = solutions[0].y

        spme = models[1]
        t_spme = solutions[1].t
        y_spme = solutions[1].y

        dfn = models[2]
        t_dfn = solutions[2].t
        y_dfn = solutions[2].y

        voltage_spm = pybamm.ProcessedVariable(
            spm.variables[variables[1]], t_dfn, y_spm, discs[0].mesh
        )
        voltage_spme = pybamm.ProcessedVariable(
            spme.variables[variables[1]], t_dfn, y_spme, discs[1].mesh
        )
        voltage_dfn = pybamm.ProcessedVariable(
            dfn.variables[variables[1]], t_dfn, y_dfn, discs[2].mesh
        )

        n = len(t)
        spm_dfn = np.sqrt(sum((voltage_dfn(t) - voltage_spm(t)) ** 2) / n)
        spme_dfn = np.sqrt(sum((voltage_dfn(t) - voltage_spme(t)) ** 2) / n)

        RMSE["SPM-DFN"].append(spm_dfn)
        RMSE["SPMe-DFN"].append(spme_dfn)

    if export_data:

        t_dfn = solutions[2].t
        t_out = np.linspace(0, t_dfn[-1], 200)
        exporter.set_output_points(t_out)

        exporter.reset_stage()

        for i, model in enumerate(models):
            exporter.set_model_solutions(model, mesh, solutions[i])

            if i == 0:
                exporter.add(variables)
            else:
                exporter.add([variables[-1]])

        exporter.export("C" + str(C_rate))

if print_RMSE:
    print("The RMSE is: \n")
    print("C-rates: ", C_rates)
    print("SPM-DFN: ", RMSE["SPM-DFN"])
    print("SPMe-DFN:", RMSE["SPMe-DFN"])

if generate_plots:
    plt.title("SPM (dotted), SPMe (dashed), DFN(solid)")
    plt.show()
