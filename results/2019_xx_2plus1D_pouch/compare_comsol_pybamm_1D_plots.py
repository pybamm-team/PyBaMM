import pybamm
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import shared
import scipy.interpolate as interp

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# increase recursion limit for large expression trees
sys.setrecursionlimit(10000)

"-----------------------------------------------------------------------------"
"Load comsol data"

C_rates = {"01": 0.1, "05": 0.5, "1": 1, "2": 2, "3": 3}
C_rate = "1"  # choose the key from the above dictionary of available results

# time-voltage (both just 1D arrays)
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/normal_mesh_{}C/voltage.csv".format(C_rate),
    # "input/comsol_results_csv/2plus1D/{}C/voltage.csv".format(C_rate),
    sep=",",
    header=None,
)
comsol_time = comsol[0].values
comsol_voltage = comsol[1].values

# volume-averaged temperature
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/normal_mesh_{}C/vol_av_T.csv".format(C_rate),
    #"input/comsol_results_csv/2plus1D/{}C/vol_av_T.csv".format(C_rate),
    sep=",",
    header=None,
)
comsol_temperature = comsol[1].values

"-----------------------------------------------------------------------------"
"Create and solve pybamm model"

# load model and geometry
pybamm.set_logging_level("INFO")
options = {
    # "current collector": "potential pair",
    # "dimensionality": 2,
    "thermal": "x-lumped"
}
pybamm_model = pybamm.lithium_ion.DFN(options)
pybamm_model.use_simplify = False
geometry = pybamm_model.default_geometry

# load parameters and process model and geometry
param = pybamm_model.default_parameter_values
param.update({"C-rate": C_rates[C_rate]})
param.process_model(pybamm_model)
param.process_geometry(geometry)

# create mesh
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 10,
    var.x_s: 10,
    var.x_p: 10,
    var.r_n: 10,
    var.r_p: 10,
    var.y: 5,
    var.z: 5,
}
mesh = pybamm.Mesh(geometry, pybamm_model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, pybamm_model.default_spatial_methods)
disc.process_model(pybamm_model)

# discharge timescale
tau = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.tau_discharge
).evaluate()

# solve model at comsol times
t_eval = comsol_time / tau
solution = pybamm_model.default_solver.solve(pybamm_model, t_eval)


"-----------------------------------------------------------------------------"
"Make Comsol 'model' for comparison"

comsol_model = pybamm.BaseModel()


def comsol_voltage_fun(t):
    return interp.interp1d(comsol_time, comsol_voltage)(t)


def comsol_temperature_fun(t):
    return interp.interp1d(comsol_time, comsol_temperature)(t)


comsol_model.variables = {
    "Terminal voltage [V]": comsol_voltage_fun,
    "Volume-averaged cell temperature [K]": comsol_temperature_fun,
}

# Process pybamm variables for which we have corresponding comsol variables
output_variables = {}
for var in comsol_model.variables.keys():
    output_variables[var] = pybamm.ProcessedVariable(
        pybamm_model.variables[var], solution.t, solution.y, mesh=mesh
    )

"-----------------------------------------------------------------------------"
"Make plots"
t_plot = comsol_time  # dimensional in seconds
shared.plot_t_var("Terminal voltage [V]", t_plot, comsol_model, output_variables, param)
shared.plot_t_var(
    "Volume-averaged cell temperature [K]",
    t_plot,
    comsol_model,
    output_variables,
    param,
)
plt.show()
