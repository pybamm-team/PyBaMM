#
# Example showing how to customize thermal boundary conditions in a pouch cell model
#
from __future__ import annotations

import numpy as np

import pybamm

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.SPM(
    {"current collector": "potential pair", "dimensionality": 2, "thermal": "x-lumped"}
)

# update parameter values, to use a spatially-varying ambient temperature and a
# spatially-varying edge heat transfer coefficient that is zero everywhere except
# at the right edge of the cell
param = model.param
L_y = param.L_y
L_z = param.L_z
parameter_values = model.default_parameter_values


def T_amb(y, z, t):
    return 300 + 20 * pybamm.sin(np.pi * y / L_y) * pybamm.sin(np.pi * z / L_z)


def h_edge(y, z):
    return pybamm.InputParameter("h_right") * (y >= L_y)


parameter_values.update(
    {
        "Current function [A]": 2 * 0.680616,
        "Ambient temperature [K]": 298,
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0,
        "Edge heat transfer coefficient [W.m-2.K-1]": h_edge,
    }
)

# create and solve simulation
var_pts = {"x_n": 4, "x_s": 4, "x_p": 4, "r_n": 4, "r_p": 4, "y": 16, "z": 16}
sim = pybamm.Simulation(model, parameter_values=parameter_values, var_pts=var_pts)
sim.build()
sim.solve([0, 600], inputs={"h_right": 5})

# plot
output_variables = [
    "Negative current collector potential [V]",
    "Positive current collector potential [V]",
    "X-averaged cell temperature [K]",
    "Voltage [V]",
]
sim.plot(output_variables, variable_limits="tight", shading="auto")
