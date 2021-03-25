#
# Example showing how to load and solve the SPMe
#

import pybamm
import numpy as np

# pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.SPMe()
# model.convert_to_format = "python"

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model for 1 hour
t_eval = np.linspace(0, 3600, 100)
solver = pybamm.CasadiSolver(mode="fast", rtol=1e-6, atol=1e-6)
solution = solver.solve(model, t_eval)
solve_time = 0
int_time = 0
for _ in range(1000):
    solution = solver.solve(model, t_eval)
    solve_time += solution.solve_time
    int_time += solution.integration_time

print(str(solve_time / 1000) + " (" + str(int_time / 1000) + ")")

# plot
plot = pybamm.QuickPlot(
    solution,
    [
        "Negative particle concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "Positive particle concentration [mol.m-3]",
        "Current [A]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        "Terminal voltage [V]",
    ],
    time_unit="seconds",
    spatial_unit="um",
)
# plot.dynamic_plot()
