#
# Compare solution of li-ion battery models when varying the number of grid points
#
import numpy as np
import pybamm
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

# choose number of points per domain (all domains will have same npts)
Npts = [30, 20, 10, 5]

# create models
models = [None] * len(Npts)
for i, npts in enumerate(Npts):
    models[i] = pybamm.lithium_ion.DFN(name="Npts = {}".format(npts))

# load parameter values and process models and geometry
param = models[0].default_parameter_values
for model in models:
    param.process_model(model)

# set mesh
meshes = [None] * len(models)

# create geometry and discretise models
var = pybamm.standard_spatial_vars
for i, model in enumerate(models):
    geometry = model.default_geometry
    param.process_geometry(geometry)
    var_pts = {
        var.x_n: Npts[i],
        var.x_s: Npts[i],
        var.x_p: Npts[i],
        var.r_n: Npts[i],
        var.r_p: Npts[i],
    }
    meshes[i] = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(meshes[i], model.default_spatial_methods)
    disc.process_model(model)

# solve model and plot voltage
solutions = [None] * len(models)
voltages = [None] * len(models)
voltage_rmse = [None] * len(models)
t_eval = np.linspace(0, 0.17, 100)
for i, model in enumerate(models):
    solutions[i] = model.default_solver.solve(model, t_eval)
    voltages[i] = pybamm.ProcessedVariable(
        model.variables["Terminal voltage [V]"],
        solutions[i].t,
        solutions[i].y,
        mesh=meshes[i],
    )(solutions[i].t)
    voltage_rmse[i] = pybamm.rmse(voltages[0], voltages[i])
    plt.plot(solutions[i].t, voltages[i], label=model.name)

for i, npts in enumerate(Npts):
    print(
        "npts = {}, solve time = {} s, Voltage RMSE = {}".format(
            npts, solutions[i].solve_time, voltage_rmse[i]
        )
    )

plt.xlabel(r"$t$")
plt.ylabel("Voltage [V]")
plt.legend()
plt.show()
