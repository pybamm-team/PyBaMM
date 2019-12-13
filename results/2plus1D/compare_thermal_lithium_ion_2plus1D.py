import pybamm
import numpy as np
import matplotlib.pyplot as plt
import sys

# set logging level and increase recursion limit
pybamm.set_logging_level("INFO")
sys.setrecursionlimit(10000)

# load models
models = [
    pybamm.lithium_ion.SPM({"thermal": "x-lumped"}, name="1D SPM (lumped)"),
    pybamm.lithium_ion.SPMe({"thermal": "x-lumped"}, name="1D SPMe (lumped)"),
    pybamm.lithium_ion.DFN({"thermal": "x-lumped"}, name="1D DFN (lumped)"),
    pybamm.lithium_ion.SPM({"thermal": "x-full"}, name="1D SPM (full)"),
    pybamm.lithium_ion.SPMe({"thermal": "x-full"}, name="1D SPMe (full)"),
    pybamm.lithium_ion.DFN({"thermal": "x-full"}, name="1D DFN (full)"),
    pybamm.lithium_ion.SPM(
        {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "xyz-lumped",
        },
        name="2+1D SPM (lumped)",
    ),
    pybamm.lithium_ion.SPMe(
        {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "xyz-lumped",
        },
        name="2+1D SPMe (lumped)",
    ),
    pybamm.lithium_ion.DFN(
        {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "xyz-lumped",
        },
        name="2+1D DFN (lumped)",
    ),
    pybamm.lithium_ion.SPM(
        {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
        },
        name="2+1D SPM (full)",
    ),
    pybamm.lithium_ion.SPMe(
        {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
        },
        name="2+1D SPMe (full)",
    ),
    pybamm.lithium_ion.DFN(
        {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
        },
        name="2+1D DFN (full)",
    ),
]

# load parameter values
param = models[0].default_parameter_values

# process models
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
temperatures = [None] * len(models)

t_eval = np.linspace(0, 1, 1000)
for i, model in enumerate(models):
    if "2+1D" in model.name:
        model.use_simplify = False  # simplifying jacobian slow for large systems
    solution = model.default_solver.solve(model, t_eval)
    solutions[i] = solution
    times[i] = solution["Time [h]"]
    voltages[i] = solution["Terminal voltage [V]"]
    temperatures[i] = solution["Volume-averaged cell temperature [K]"]

# plot terminal voltage and temperature
t = np.linspace(0, solution.t[-1], 100)
plt.subplot(121)
for i, model in enumerate(models):
    plt.plot(times[i](t), voltages[i](t), label=model.name)
plt.xlabel("Time [h]")
plt.ylabel("Terminal voltage [V]")
plt.legend()
plt.subplot(122)
for i, model in enumerate(models):
    plt.plot(times[i](t), temperatures[i](t), label=model.name)
plt.xlabel("Time [h]")
plt.ylabel("Temperature [K]")
plt.tight_layout()
plt.show()
