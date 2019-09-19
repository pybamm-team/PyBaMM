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
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.2, 100)
solution = model.default_solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(model, mesh, solution)
plot.dynamic_plot()
