#
# Simulate drive cycle loaded from csv file
#
import pybamm
import numpy as np

# load model
pybamm.set_logging_level("INFO")
model = pybamm.lithium_ion.SPMe()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param["Current function"] = "[current data]US06"
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# simulate US06 drive cycle
tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_eval = np.linspace(0, 600 / tau, 600)

# need to increase max solver steps if solving DAEs along with an erratic drive cycle
solver = pybamm.CasadiSolver()
if isinstance(solver, pybamm.DaeSolver):
    solver.max_steps = 10000

solution = solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(model, mesh, solution)
plot.dynamic_plot()
