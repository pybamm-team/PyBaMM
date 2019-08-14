#
# Figure 5: Computation times and errors for changing grid points
#
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import shared

generate_plots = True
export_data = False

C_rates = [0.1, 0.5, 1, 2, 3]
colour = {0.1: "r", 0.5: "b", 1: "g", 2: "m", 3: "y"}
t_eval = np.linspace(0, 0.17, 100)

points = [5, 10, 20, 30]
truth_pts = 30
pts_color = {5: "r", 10: "b", 20: "g", 30: "m"}


# calculate the truth voltage (taken to be high number of pts on dfn)
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: truth_pts,
    var.x_s: truth_pts,
    var.x_p: truth_pts,
    var.r_n: truth_pts,
    var.r_p: truth_pts,
}
model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

truth = {}
t_out = {}
for C_rate in C_rates:
    param["Typical current [A]"] = (
        C_rate * 24 * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
    )
    param.update_model(model, disc)
    true_sol = model.default_solver.solve(model, t_eval)

    t, y = true_sol.t, true_sol.y
    t_out[C_rate] = np.linspace(0, t[-1], 100)
    truth[C_rate] = pybamm.ProcessedVariable(
        model.variables["Terminal voltage [V]"], t, y, mesh
    )(t_out[C_rate])

#####################


errors = {"spme": {}, "spm": {}, "dfn": {}}
difference = {"spme": {}, "spm": {}}
times = {"spme": {}, "spm": {}, "dfn": {}}


# vary electrode points
for pts in points:

    spm = pybamm.lithium_ion.SPM()
    spme = pybamm.lithium_ion.SPMe()
    dfn = pybamm.lithium_ion.DFN()
    models = shared.ModelGroup(spme, spm, dfn)
    models.process_parameters()

    var_pts = {var.x_n: pts, var.x_s: pts, var.x_p: pts, var.r_n: pts, var.r_p: pts}

    models.discretise(var_pts)

    errors["spme"][pts] = []
    errors["spm"][pts] = []
    errors["dfn"][pts] = []

    difference["spme"][pts] = []
    difference["spm"][pts] = []

    times["spme"][pts] = []
    times["spm"][pts] = []
    times["dfn"][pts] = []

    for C_rate in C_rates:
        update_parameters = {
            "Typical current [A]": C_rate
            * 24
            * models.parameters.process_symbol(
                pybamm.geometric_parameters.A_cc
            ).evaluate()
        }
        models.solve(t_eval, update_parameters)

        processed_variables = models.process_variables(["Terminal voltage [V]"])

        def rmse(compare, value):
            return np.sqrt(sum((compare - value) ** 2) / len(value))

        errors["spme"][pts].append(
            rmse(
                truth[C_rate],
                processed_variables[spme]["Terminal voltage [V]"](t_out[C_rate]),
            )
        )
        errors["spm"][pts].append(
            rmse(
                truth[C_rate],
                processed_variables[spm]["Terminal voltage [V]"](t_out[C_rate]),
            )
        )
        errors["dfn"][pts].append(
            rmse(
                truth[C_rate],
                processed_variables[dfn]["Terminal voltage [V]"](t_out[C_rate]),
            )
        )

        difference["spme"][pts].append(
            rmse(
                processed_variables[dfn]["Terminal voltage [V]"](t_out[C_rate]),
                processed_variables[spme]["Terminal voltage [V]"](t_out[C_rate]),
            )
        )
        difference["spm"][pts].append(
            rmse(
                processed_variables[dfn]["Terminal voltage [V]"](t_out[C_rate]),
                processed_variables[spm]["Terminal voltage [V]"](t_out[C_rate]),
            )
        )

        times["spme"][pts].append(models.times[0])
        times["spm"][pts].append(models.times[1])
        times["dfn"][pts].append(models.times[2])

    plt.subplot(2, 1, 1)
    plt.plot(C_rates, errors["spm"][pts], linestyle=":", color=pts_color[pts])
    plt.plot(C_rates, errors["spme"][pts], linestyle="--", color=pts_color[pts])
    plt.plot(C_rates, errors["dfn"][pts], linestyle="-", color=pts_color[pts])
    plt.xlabel("C-rate")
    plt.ylabel("Voltage error [V]")


average_times = {"spme": {}, "spm": {}, "dfn": {}}
for pts in points:
    for model, pt in times.items():
        for p, crates in pt.items():
            average_times[model][p] = np.mean(crates)

plt.subplot(2, 1, 2)
plt.plot(points, average_times["spm"], linestyle=":")
plt.plot(points, average_times["spme"], linestyle="--")
plt.plot(points, average_times["dfn"], linestyle="-")
plt.xlabel("points")
plt.ylabel("computation time")

plt.show()
