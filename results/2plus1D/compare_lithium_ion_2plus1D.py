import pybamm
import numpy as np
import matplotlib.pyplot as plt
import sys

# set logging level and increase recursion limit
pybamm.set_logging_level("INFO")
sys.setrecursionlimit(10000)

# load models
models = [
    pybamm.lithium_ion.SPM(name="1D SPM"),
    pybamm.lithium_ion.SPMe(name="1D SPMe"),
    pybamm.lithium_ion.DFN(name="1D DFN"),
    pybamm.lithium_ion.SPM(
        {"current collector": "potential pair", "dimensionality": 2}, name="2+1D SPM"
    ),
    pybamm.lithium_ion.SPMe(
        {"current collector": "potential pair", "dimensionality": 2}, name="2+1D SPMe"
    ),
    pybamm.lithium_ion.DFN(
        {"current collector": "potential pair", "dimensionality": 2}, name="2+1D DFN"
    ),
]

# load parameter values
param = models[0].default_parameter_values
# adjust current to correspond to a typical current density of C_rate * 24 [A.m-2]
C_rate = 1
param["Typical current [A]"] = (
    C_rate * 24 * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
)
# make current collectors not so conductive, just for illustrative purposes
param["Negative current collector conductivity [S.m-1]"] = 5.96e6
param["Positive current collector conductivity [S.m-1]"] = 3.55e6

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

# plot terminal voltage
t = np.linspace(0, solution.t[-1], 100)
for i, model in enumerate(models):
    plt.plot(times[i](t), voltages[i](t), label=model.name)
plt.xlabel("Time [h]")
plt.ylabel("Terminal voltage [V]")
plt.legend()
# add C-rate, delta, and alpha to title
delta = param.process_symbol(pybamm.standard_parameters_lithium_ion.delta).evaluate()
alpha = param.process_symbol(pybamm.standard_parameters_lithium_ion.alpha).evaluate()
plt.title(
    r"C-rate = {:3d}, $\alpha$ = {:.6f} , $\delta$ = {:.6f}".format(
        C_rate, alpha, delta
    )
)
# save and show
file_name = "discharge_curve_2plus1D_comparison.eps"
plt.savefig(file_name, format="eps", dpi=1000)
plt.show()
