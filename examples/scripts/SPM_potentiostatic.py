import pybamm
import numpy as np

# load model
model = pybamm.lithium_ion.SPM_Potentiostatic()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
set_of_parameters = pybamm.standard_parameters_lithium_ion
OCV_init = param.process_symbol(set_of_parameters.U_p(set_of_parameters.c_p_init) - set_of_parameters.U_n(set_of_parameters.c_n_init)).evaluate(0, 0)
param.update({"Applied voltage": OCV_init})  # add parameter for local V
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
plot = pybamm.QuickPlot(model, mesh, solver)
plot.dynamic_plot()
