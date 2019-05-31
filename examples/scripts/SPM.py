import pybamm
import numpy as np

# load (2+1D) SPM model
options = {"bc_options": {"dimensionality": 2}}
model = pybamm.lithium_ion.SPM(options)

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
solver = model.default_solver
t_eval = np.linspace(0, 2, 100)
solver.solve(model, t_eval)

# plot
#plot = pybamm.QuickPlot(model, mesh, solver)
#plot.dynamic_plot()
