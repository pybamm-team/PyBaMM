#
# Example showing how to customize thermal boundary conditions in a pouch cell model
#
import numpy as np
import pybamm

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.SPM(
    {"current collector": "potential pair", "dimensionality": 2}, name="2+1D SPM"
)

# update parameter values, to use:
# 1) a spatially-varying ambient temperature
param = model.param
L_y = param.L_y
L_z = param.L_z
parameter_values = model.default_parameter_values


def T_amb(y, z, t):
    return 300 + 20 * pybamm.sin(np.pi * y / L_y) * pybamm.sin(np.pi * z / L_z)


parameter_values.update({"Ambient temperature [K]": T_amb})

# create and solve simulation
var_pts = {"x_n": 4, "x_s": 4, "x_p": 4, "r_n": 4, "r_p": 4, "y": 16, "z": 16}
sim = pybamm.Simulation(model, parameter_values=parameter_values, var_pts=var_pts)
sim.build()
sim.solve([0, 3600])

# plot
output_variables = [
    "Negative current collector potential [V]",
    "Positive current collector potential [V]",
    "X-averaged cell temperature [K]",
    "Voltage [V]",
]
sim.plot(output_variables)
