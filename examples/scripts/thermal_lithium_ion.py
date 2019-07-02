import pybamm
import numpy as np

# load model
options = {"thermal": "full"}
full_thermal_model = pybamm.lithium_ion.DFN(options)

options = {"thermal": "lumped"}
lumped_thermal_model = pybamm.lithium_ion.DFN(options)

models = [full_thermal_model, lumped_thermal_model]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param.update({"Typical current [A]": 0.3})
for model in models:
    param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}

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
output_variables = ["Electrolyte concentration", "Cell temperature [K]"]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)
plot.dynamic_plot()
