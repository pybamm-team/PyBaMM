import pybamm
import numpy as np
import os
import pickle
import scipy.interpolate as interp
import matplotlib.pyplot as plt

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

"-----------------------------------------------------------------------------"
"Pick Crate and load comsol data"

# Crate
# NOTE: the results in pybamm stop when a voltage cutoff is reached, so
# for higher C-rate the pybamm solution may stop before the comsol solution
Crate = 1
sigma = 1 * 8000

# load the comsol results
comsol_voltages = pickle.load(
    open("results/2019_08_sulzer_thesis/compare_comsol/comsol_voltages.pickle", "rb")
)

"-----------------------------------------------------------------------------"
"Create and solve pybamm model"

# load model and geometry
pybamm.set_logging_level("INFO")
pybamm_model = pybamm.lead_acid.NewmanTiedemann(
    {
        "surface form": "algebraic",
        "current collector": "potential pair",
        "dimensionality": 1,
    }
)
geometry = pybamm_model.default_geometry

# load parameters and process model and geometry
param = pybamm_model.default_parameter_values
param["Typical current [A]"] = 17 * Crate
param["Positive electrode conductivity [S.m-1]"] = sigma
# Change the t_plus function to agree with Comsol
param["Darken thermodynamic factor"] = np.ones_like
param["MacInnes t_plus function"] = lambda x: 1 - 2 * x
param.process_model(pybamm_model)
param.process_geometry(geometry)

# create mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.z: 5}
mesh = pybamm.Mesh(geometry, pybamm_model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, pybamm_model.default_spatial_methods)
disc.process_model(pybamm_model)

# discharge timescale
tau = param.process_symbol(pybamm.standard_parameters_lead_acid.tau_discharge)

# solve model at comsol times
comsol_t = comsol_voltages[Crate][sigma][2][0]
time = comsol_t / tau.evaluate(0)

solution = pybamm_model.default_solver.solve(pybamm_model, time)

comsol_voltage = interp.interp1d(comsol_t, comsol_voltages[Crate][sigma][2][1])
# Create comsol model with dictionary of Matrix variables
comsol_model = pybamm.BaseModel()
comsol_model.variables = {
    "Terminal voltage [V]": pybamm.Function(comsol_voltage, pybamm.t * tau)
}

# plot
plot = pybamm.QuickPlot(
    [pybamm_model, comsol_model],
    mesh,
    [solution, solution],
    output_variables=comsol_model.variables.keys(),
    labels=["PyBaMM", "Comsol"],
)
plot.dynamic_plot()
