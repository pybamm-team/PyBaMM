import pybamm
import numpy as np
import os

# load model
pybamm.set_logging_level("DEBUG")
model = pybamm.lithium_ion.SPMe()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param["Current function"] = os.path.join(
    os.getcwd(), "pybamm", "parameters", "standard_current_functions", "car_current.py"
)
param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# simulate car current for 30 minutes
tau = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.tau_discharge
).evaluate(0)
t_eval = np.linspace(0, 1800 / tau, 100)

# need to increase max solver steps if solving DAEs along with an erratic drive cycle
solver = model.default_solver
if isinstance(solver, pybamm.DaeSolver):
    solver.max_steps = 10000

solution = solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(model, mesh, solution)
plot.dynamic_plot()
