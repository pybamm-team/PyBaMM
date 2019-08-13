#
# Figure 3: Overpotential errors
#
import pybamm
import numpy as np
import matplotlib.pyplot as plt

generate_plots = True
print_RMSE = True
export_data = True


models = [pybamm.lithium_ion.SPMe(), pybamm.lithium_ion.DFN()]

param = models[0].default_parameter_values
C_rate = 3
param["Typical current [A]"] = (
    C_rate * 24 * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
)

for model in models:
    param.process_model(model)

var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 20, var.x_p: 30, var.r_n: 15, var.r_p: 15}
var_pts = {var.x_n: 6, var.x_s: 6, var.x_p: 6, var.r_n: 5, var.r_p: 5}

discs = []
for model in models:
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    discs.append(disc)


solutions = [None] * len(models)
t_eval = np.linspace(0, 0.17, 100)
for i, model in enumerate(models):
    param.update_model(model, discs[i])
    solutions[i] = model.default_solver.solve(model, t_eval)


output_variables = [
    "X-averaged open circuit voltage [V]",
    "X-averaged reaction overpotential [V]",
    "X-averaged electrolyte overpotential [V]",
]

if generate_plots:

    spme, dfn = models
    spme_disc, dfn_disc = discs
    t_spme, y_spme = solutions[0].t, solutions[0].y
    t_dfn, y_dfn = solutions[1].t, solutions[1].y

    t_eval = np.linspace(0, t_dfn[-1], 200)

    discharge_cap = pybamm.ProcessedVariable(
        dfn.variables["Discharge capacity [A.h.m-2]"], t_dfn, y_dfn, dfn_disc.mesh
    )(t_eval)

    spme_output_variables = {var: spme.variables[var] for var in output_variables}
    dfn_output_variables = {var: dfn.variables[var] for var in output_variables}

    spme_post_process_variables = pybamm.post_process_variables(
        spme_output_variables, t_spme, y_spme, spme_disc.mesh
    )
    dfn_post_process_variables = pybamm.post_process_variables(
        dfn_output_variables, t_dfn, y_dfn, dfn_disc.mesh
    )

    ocp_diff = np.abs(
        spme_post_process_variables[output_variables[0]](t_eval)
        - dfn_post_process_variables[output_variables[0]](t_eval)
    )
    eta_r_diff = np.abs(
        spme_post_process_variables[output_variables[1]](t_eval)
        - dfn_post_process_variables[output_variables[1]](t_eval)
    )
    eta_e_diff = np.abs(
        spme_post_process_variables[output_variables[2]](t_eval)
        - dfn_post_process_variables[output_variables[2]](t_eval)
    )

    plt.fill_between(discharge_cap, ocp_diff, label="OCV")
    plt.fill_between(
        discharge_cap, ocp_diff, ocp_diff + eta_r_diff, lw=2, color="g", label="eta_r"
    )
    plt.fill_between(
        discharge_cap,
        ocp_diff + eta_r_diff,
        ocp_diff + eta_r_diff + eta_e_diff,
        lw=2,
        color="r",
        label="eta_e",
    )
    plt.xlabel("Discharge capacity [A.h.m-2]", fontsize=15)
    plt.ylabel("Voltage error [V]", fontsize=15)
    plt.legend(fontsize=15, loc=2)
    plt.show()

if export_data:

    dir_path = "results/2019_08_asymptotic_spme/data/figure_3"
    exporter = pybamm.ExportCSV(dir_path)

    t_dfn = solutions[1].t
    t_out = np.linspace(0, t_dfn[-1], 200)
    exporter.set_output_points(t_out)

    exporter.set_model_solutions(models[0], mesh, solutions[0])
    exporter.add(["Discharge capacity [A.h.m-2]"])
    exporter.set_model_solutions(models, mesh, solutions)

    exporter.add_error(output_variables, truth_index=1)

    exporter.export("figure_3")
