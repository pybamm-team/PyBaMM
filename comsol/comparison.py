import pybamm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load the comsol voltage data
comsol = pd.read_csv("comsol/Voltage.csv", sep=",", header=None)

comsol_time = comsol[0].values
comsol_voltage = comsol[1].values

# load model and geometry
model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry

# load parameters and process model and geometry
param = model.default_parameter_values

# update C_rate
# C_rate = 1
# param["Typical current density"] = 24 * C_rate

param.process_model(model)
param.process_geometry(geometry)

# create mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 10, var.x_p: 30, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
solver = model.default_solver
time = np.linspace(0, 1.2, 100)
solver.solve(model, time)  # use time from the comsol simulation (doesn't really matter)

# extract the voltage
voltage = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solver.t, solver.y, mesh=mesh
)

# plt.plot(time, comsol_voltage)
tau_d = 1  # get tau_d here from params
plt.plot(comsol_time, comsol_voltage, solver.t * tau_d, voltage(solver.t), ":b")
plt.legend(["comsol", "pybamm"])
plt.show()
