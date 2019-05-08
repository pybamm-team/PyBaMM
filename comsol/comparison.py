import pybamm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load the comsol voltage data
comsol = pd.read_csv("comsol/Voltage.csv", sep=",", header=None)

time = comsol[0].values
comsol_voltage = comsol[1].values


# load model and geometry
model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry

# load parameters and process model and geometry
param = model.default_parameter_values

# update C_rate
C_rate = 1
param["Typical current density"] = 24 * C_rate

param.process_model(model)
param.process_geometry(geometry)

# convert time to dimensionless form
tau = pybamm.standard_parameters_lithium_ion.tau_discharge
tau_eval = param.process_symbol(tau).evaluate(0, 0)

time = time / tau_eval

# create mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 3, var.x_s: 3, var.x_p: 3, var.r_n: 6, var.r_p: 6}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
solver = model.default_solver
t = np.linspace(0, 2, 500)
solver.solve(model, t)  # use time from the comsol simulation (doesn't really matter)

# extract the voltage
voltage = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solver.t, solver.y, mesh=mesh
)

voltage_sol = voltage(solver.t)

# dimensional time
time = time * tau_eval / 60 / 60
t = solver.t * tau_eval / 60 / 60

plt.plot(time, comsol_voltage, "r")
plt.plot(t, voltage_sol, ":b")
plt.legend(["comsol", "pybamm"])
plt.show()

quick_plot = pybamm.QuickPlot(model, param, mesh, solver)

quick_plot.dynamic_plot()
