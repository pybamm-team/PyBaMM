#
# Compares the full and lumped thermal models for a single layer Li-ion cell
#

import pybamm
import numpy as np

# load model
pybamm.set_logging_level("INFO")

options = {"thermal": "x-full"}
full_thermal_model = pybamm.lithium_ion.SPMe(options)

options = {"thermal": "x-lumped"}
lumped_thermal_model = pybamm.lithium_ion.SPMe(options)

models = [full_thermal_model, lumped_thermal_model]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param.update({"Heat transfer coefficient [W.m-2.K-1]": 0.1})

for model in models:
    param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 10, var.r_p: 10}

# discretise models
for model in models:
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 0.17, 100)
for i, model in enumerate(models):
    solution = model.default_solver.solve(model, t_eval)
    solutions[i] = solution

# plot
output_variables = [
    "Terminal voltage [V]",
    "X-averaged cell temperature [K]",
    "Cell temperature [K]",
]
labels = ["Full thermal model", "Lumped thermal model"]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables, labels)
plot.dynamic_plot()
