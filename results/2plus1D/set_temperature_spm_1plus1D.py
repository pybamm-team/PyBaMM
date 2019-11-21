#
# Example of 1+1D SPM where the temperature can be set by the user
#

import pybamm
import numpy as np

# set logging level
pybamm.set_logging_level("INFO")

model_options = {
    "current collector": "potential pair",
    "dimensionality": 1,
    "thermal": "x-lumped",
    "external submodels": ["thermal"],
}
model = pybamm.lithium_ion.SPMe(model_options)

var = pybamm.standard_spatial_vars
z_pts = 20
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5, var.z: z_pts}

sim = pybamm.Simulation(model, var_pts=var_pts, C_rate=2)

# Set the temperature (in dimensionless form)
# T_av = np.linspace(0, 1, z_pts)[:, np.newaxis]

z = np.linspace(0, 1, z_pts)
t_eval = np.linspace(0, 0.13, 50)
# step through the solver, setting the temperature at each timestep
for i in np.arange(1, len(t_eval) - 1):
    dt = t_eval[i + 1] - t_eval[i]
    T_av = (np.sin(2 * np.pi * z) * np.sin(2 * np.pi * 100 * t_eval[i]))[
        :, np.newaxis
    ]
    external_variables = {"X-averaged cell temperature": T_av}
    sim.step(dt, external_variables=external_variables)

sim.plot(
    [
        "Terminal voltage [V]",
        "X-averaged total heating [W.m-3]",
        "X-averaged cell temperature [K]",
        "X-averaged negative particle surface concentration [mol.m-3]",
        "X-averaged positive particle surface concentration [mol.m-3]",
        "Negative current collector potential [V]",
        "Positive current collector potential [V]",
        "Local voltage [V]",
    ]
)

