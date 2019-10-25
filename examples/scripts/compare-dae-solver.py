import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.DFN()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars

var_pts = {var.x_n: 60, var.x_s: 100, var.x_p: 60, var.r_n: 50, var.r_p: 50}
# var_pts = model.default_var_pts
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.2, 100)

klu_sol = pybamm.KLU(atol=1e-8, rtol=1e-8).solve(model, t_eval)
scikits_sol = pybamm.ScikitsDaeSolver(atol=1e-8, rtol=1e-8).solve(model, t_eval)

# plot
models = [model, model]
solutions = [scikits_sol, klu_sol]
plot = pybamm.QuickPlot(models, mesh, solutions)
plot.dynamic_plot()
