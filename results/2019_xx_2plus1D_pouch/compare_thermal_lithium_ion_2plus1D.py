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
    pybamm.lithium_ion.DFN({"thermal": "x-lumped"}, name="1D DFN (lumped)"),
    pybamm.lithium_ion.SPM(
        {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
        },
        name="2+1D SPM (full)",
    ),
]

# load parameter values
param = models[0].default_parameter_values
# adjust current to correspond to a typical current density of 24 [A.m-2]
C_rate = 1
param["Typical current [A]"] = (
    C_rate * 24 * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
)

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
        var.x_n: 10,
        var.x_s: 10,
        var.x_p: 10,
        var.r_n: 10,
        var.r_p: 10,
        var.y: 10,
        var.z: 10,
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
    times[i] = pybamm.ProcessedVariable(
        model.variables["Time [h]"], solution.t, solution.y
    )
    voltages[i] = pybamm.ProcessedVariable(
        model.variables["Terminal voltage [V]"], solution.t, solution.y, mesh=meshes[i]
    )
    temperatures[i] = pybamm.ProcessedVariable(
        model.variables["Volume-averaged cell temperature [K]"],
        solution.t,
        solution.y,
        mesh=meshes[i],
    )

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
