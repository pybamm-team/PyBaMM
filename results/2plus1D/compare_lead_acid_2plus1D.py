import pybamm
import numpy as np
import matplotlib.pyplot as plt
import sys

# set logging level and increase recursion limit
pybamm.set_logging_level("INFO")
sys.setrecursionlimit(10000)

# load models
models = [
    pybamm.lead_acid.Full(name="1D Full"),
    pybamm.lead_acid.Composite(name="1D composite"),
    pybamm.lead_acid.LOQS(name="1D LOQS"),
    pybamm.lead_acid.Full(
        {"current collector": "potential pair", "dimensionality": 2}, name="2+1D Full"
    ),
    pybamm.lead_acid.Composite(
        {"current collector": "potential pair", "dimensionality": 2},
        name="2+1D composite",
    ),
    pybamm.lead_acid.LOQS(
        {"current collector": "potential pair", "dimensionality": 2}, name="2+1D LOQS"
    ),
]

# load parameter values and process models
param = models[0].default_parameter_values
for model in models:
    param.process_model(model)

# process geometry and discretise models
meshes = [None] * len(models)
for i, model in enumerate(models):
    geometry = model.default_geometry
    param.process_geometry(geometry)
    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: 5,
        var.x_s: 5,
        var.x_p: 5,
        var.r_n: 5,
        var.r_p: 5,
        var.y: 5,
        var.z: 5,
    }
    meshes[i] = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(meshes[i], model.default_spatial_methods)
    disc.process_model(model)

# solve models and process time and voltage for plotting on different meshes
solutions = [None] * len(models)
times = [None] * len(models)
voltages = [None] * len(models)
t_eval = np.linspace(0, 1, 1000)
for i, model in enumerate(models):
    if "2+1D" in model.name:
        model.use_simplify = False  # simplifying jacobian slow for large systems
    solution = model.default_solver.solve(model, t_eval)
    solutions[i] = solution
    times[i] = solution["Time [h]"]
    voltages[i] = solution["Terminal voltage [V]"]

# plot terminal voltage
t = np.linspace(0, solution.t[-1], 100)
for i, model in enumerate(models):
    plt.plot(times[i](t), voltages[i](t), lw=2, label=model.name)
plt.xlabel("Time [h]", fontsize=15)
plt.ylabel("Terminal voltage [V]", fontsize=15)
plt.legend(fontsize=15)
plt.show()
