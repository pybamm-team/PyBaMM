#
# Example showing how to load and solve the SPMe
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.SPM()
model.convert_to_format = "python"

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}

mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
rhs = pybamm.get_julia_function(model.concatenated_rhs.simplify())

from julia import Main

rhs_eval = Main.eval(rhs)
n = 1


# # solve model for 1 hour
# t_eval = np.linspace(0, 3600, 100)
# solution = model.default_solver.solve(model, t_eval)

# # plot
# plot = pybamm.QuickPlot(
#     solution,
#     [
#         "Negative particle concentration [mol.m-3]",
#         "Electrolyte concentration [mol.m-3]",
#         "Positive particle concentration [mol.m-3]",
#         "Current [A]",
#         "Negative electrode potential [V]",
#         "Electrolyte potential [V]",
#         "Positive electrode potential [V]",
#         "Terminal voltage [V]",
#     ],
#     time_unit="seconds",
#     spatial_unit="um",
# )
# plot.dynamic_plot()
