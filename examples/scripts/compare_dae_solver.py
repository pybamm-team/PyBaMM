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
var_pts = {"x_n": 50, "x_s": 50, "x_p": 50, "r_n": 20, "r_p": 20}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 3600, 100)

casadi_sol = pybamm.CasadiSolver(atol=1e-8, rtol=1e-8).solve(model, t_eval)
solutions = [casadi_sol]

klu_sol = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8).solve(model, t_eval)
solutions.append(klu_sol)

# plot
plot = pybamm.QuickPlot(solutions)
plot.dynamic_plot()
