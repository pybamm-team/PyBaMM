#
# Example showing how to load and solve the DFN for the half cell
#

import pybamm
import numpy as np
from pybamm.geometry import half_cell_spatial_vars

pybamm.set_logging_level("INFO")

# load model
options = {"working electrode": "negative"}
model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)

# create geometry
geometry = model.default_geometry

# load parameter values
chemistry = pybamm.parameter_sets.Chen2020
param = pybamm.ParameterValues(chemistry=chemistry)

# add lithium counter electrode parameter values
param.update(
    {
        "Lithium counter electrode exchange-current density [A.m-2]": 12.6,
        "Lithium counter electrode conductivity [S.m-1]": 1.0776e7,
        "Lithium counter electrode thickness [m]": 250e-6,
    },
    check_already_exists=False,
)

param["Initial concentration in negative electrode [mol.m-3]"] = 1000
param["Current function [A]"] = 2.5

# process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# set mesh
# var = half_cell_spatial_vars
# var_pts = {
#     var.x_Li: 20,
#     var.x_s: 200,
#     var.x_w: 200,
#     var.r_w: 30,
#     var.y: 10,
#     var.z: 10,
# }

var_pts = model.default_var_pts
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 7200, 1000)
solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
solution = solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(
    solution,
    [
        "Working particle surface concentration [mol.m-3]",
        # "Working particle concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "Current [A]",
        "Working electrode potential [V]",
        "Electrolyte potential [V]",
        "Total electrolyte concentration",
        "Total lithium in working electrode [mol]",
        "Working electrode open circuit potential [V]",
        ["Terminal voltage [V]", "Voltage drop in the cell [V]"],
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
