#
# Example showing how to load and solve the DFN
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
options = {"working electrode": "cathode"}
model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)

Crate = 0.5
tpulse = 360
trest = 3600
Npulse = np.ceil(3600 / (tpulse * Crate))
tend = (tpulse + trest) * Npulse

def GITT_current(Crate, tpulse, trest):
    def current(t):
        # return Crate * pybamm.EqualHeaviside(t % (tpulse + trest), tpulse)
        return Crate * pybamm.EqualHeaviside(t, tpulse)

    return current


# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
chemistry = pybamm.parameter_sets.Chen2020
param = pybamm.ParameterValues(chemistry=chemistry)
param.update(
    {
        "Lithium counter electrode exchange-current density [A.m-2]": 12.6,
        "Lithium counter electrode conductivity [S.m-1]": 1.0776e7,
        "Lithium counter electrode thickness [m]": 250e-6,
    },
    check_already_exists=False,
)
param["Current function [A]"] = GITT_current(Crate, tpulse, trest)
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
# t_eval = np.linspace(0, tend, tend // 10)
t_eval = np.linspace(0, 3800, 1000)
solver = pybamm.CasadiSolver(mode="fast", atol=1e-6, rtol=1e-3)
solution = solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(
    solution,
    [
        "Negative particle surface concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "Positive particle surface concentration [mol.m-3]",
        "Current [A]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        ["Terminal voltage [V]", "Voltage drop [V]"],
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
