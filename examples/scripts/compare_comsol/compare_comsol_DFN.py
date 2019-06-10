import pybamm
import os
import pickle

# change working directory to the root of pybamm
os.chdir(pybamm.__path__[0] + "/..")

"-----------------------------------------------------------------------------"
"Pick C_rate and load comsol data"

# C_rate
# NOTE: the results in pybamm stop when a voltage cutoff is reached, so
# for higher C-rate the pybamm solution may stop before the comsol solution
C_rates = {"01": 0.1, "05": 0.5, "1": 1, "2": 2, "3": 3}
C_rate = "1"  # choose the key from the above dictionary of available results

# load the comsol results
comsol_variables = pickle.load(
    open("input/comsol_results/comsol_{}C.pickle".format(C_rate), "rb")
)

"-----------------------------------------------------------------------------"
"Create and solve pybamm model"

# load model and geometry
model = pybamm.lithium_ion.SPM()
geometry = model.default_geometry

# load parameters and process model and geometry
param = model.default_parameter_values
param["Electrode depth [m]"] = 1
param["Electrode height [m]"] = 1
param["Typical current [A]"] = 24 * C_rates[C_rate]
param.process_model(model)
param.process_geometry(geometry)

# create mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 31, var.x_s: 11, var.x_p: 31, var.r_n: 11, var.r_p: 11}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# discharge timescale
tau = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.tau_discharge
).evaluate(0, 0)

# solve model at comsol times
time = comsol_variables["time"] / tau
solution = model.default_solver.solve(model, time)

"-----------------------------------------------------------------------------"
"Make Comsol 'model' for comparison"

whole_cell = ["negative electrode", "separator", "positive electrode"]
comsol_model = pybamm.BaseModel()
import ipdb

ipdb.set_trace()
comsol_model.variables = {
    "Negative particle surface concentration [mol.m-3]": pybamm.Matrix(
        comsol_variables["c_n_surf"], domain=["negative electrode"]
    ),
    "Electrolyte concentration [mol.m-3]": pybamm.Matrix(
        comsol_variables["c_e"], domain=whole_cell
    ),
    "Positive particle surface concentration [mol.m-3]": pybamm.Matrix(
        comsol_variables["c_p_surf"], domain=["positive electrode"]
    ),
    "Typical current [A]": param["Typical current [A]"],
    "Negative electrode potential [V]": pybamm.Matrix(
        comsol_variables["phi_n"], domain=["negative electrode"]
    ),
    "Electrolyte potential [V]": pybamm.Matrix(
        comsol_variables["phi_e"], domain=whole_cell
    ),
    "Positive electrode potential [V]": pybamm.Matrix(
        comsol_variables["phi_p"], domain=["positive electrode"]
    ),
    "Terminal voltage [V]": pybamm.Matrix(comsol_variables["voltage"]),
}

# plot
models = [model, comsol_model]
solutions = [solution, solution]
plot = pybamm.QuickPlot(models, mesh, solutions, comsol_model.variables.keys())
plot.dynamic_plot()
