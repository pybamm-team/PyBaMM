#
# Compare solution of li-ion battery models when changing solver tolerances
#
import numpy as np
import pybamm

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.DFN()


# process and discretise
param = model.default_parameter_values
param.process_model(model)
geometry = model.default_geometry
param.process_geometry(geometry)
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# tolerances (rtol, atol)
tols = [[1e-8, 1e-8], [1e-6, 1e-6], [1e-3, 1e-6], [1e-3, 1e-3]]

# solve model
solutions = [None] * len(tols)
voltages = [None] * len(tols)
voltage_rmse = [None] * len(tols)
labels = [None] * len(tols)
t_eval = np.linspace(0, 0.25, 100)
for i, tol in enumerate(tols):
    solver = pybamm.ScikitsDaeSolver(rtol=tol[0], atol=tol[1])
    solutions[i] = solver.solve(model, t_eval)
    voltages[i] = pybamm.ProcessedVariable(
        model.variables["Terminal voltage [V]"],
        solutions[i].t,
        solutions[i].y,
        mesh=mesh,
    )(solutions[i].t)
    voltage_rmse[i] = pybamm.rmse(voltages[0], voltages[i])
    labels[i] = "rtol = {}, atol = {}".format(tol[0], tol[1])

# print RMSE voltage errors vs tighest tolerance
for i, tol in enumerate(tols):
    print(
        "rtol = {}, atol = {}, solve time = {} s, Voltage RMSE = {}".format(
            tol[0], tol[1], solutions[i].solve_time, voltage_rmse[i]
        )
    )
# plot
plot = pybamm.QuickPlot([model] * len(solutions), mesh, solutions, labels=labels)
plot.dynamic_plot()
