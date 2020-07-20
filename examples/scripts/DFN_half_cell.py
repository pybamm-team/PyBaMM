#
# Example showing how to load and solve the DFN
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")


# load model
model = pybamm.lithium_ion.BasicDFNHalfCell()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
Crate = 0.1
chemistry = pybamm.parameter_sets.Chen2020
param = pybamm.ParameterValues(chemistry=chemistry)
param["Current function [A]"] = Crate * 5
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
t_eval = np.linspace(0, 3600, 100)
solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-3)
solution = solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(
    solution,
    [
        "Negative particle surface concentration",
        "Electrolyte concentration",
        "Positive particle surface concentration",
        "Current [A]",
        "Negative electrode potential",
        "Electrolyte potential",
        "Positive electrode potential",
        "Terminal voltage [V]",
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
