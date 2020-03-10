#
# Solve the DFN with a varying ambient temperature
#

import pybamm
import numpy as np

pybamm.set_logging_level("DEBUG")


# load model
options = {"thermal": "x-lumped"}
model = pybamm.lithium_ion.DFN(options)

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry


def ambient_temperature(t):
    T = 5 # degree temperature change per hour
    return 300 + t * T / 3600 


param = model.default_parameter_values
param.update(
    {"Ambient temperature [K]": ambient_temperature}, check_already_exists=False
)
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 3600 / 2, 100)
solver = pybamm.CasadiSolver(mode="fast")
solver.rtol = 1e-3
solver.atol = 1e-6
solution = solver.solve(model, t_eval)

# plot
output_variables = [
    "Negative particle surface concentration",
    "Electrolyte concentration",
    "Positive particle surface concentration",
    "Current [A]",
    "Negative electrode potential [V]",
    "Electrolyte potential [V]",
    "Positive electrode potential [V]",
    "Terminal voltage [V]",
    "X-averaged cell temperature [K]", 
    "Ambient temperature [K]"
]
plot = pybamm.QuickPlot(
    solution, output_variables
)
plot.dynamic_plot()
