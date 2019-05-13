import pybamm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"-----------------------------------------------------------------------------"
"Pick C_rate and load comsol data"

# C_rate
C_rate = 1

# load the comsol data
comsol = pd.read_csv("comsol/Voltage_C{}.csv".format(C_rate), sep=",", header=None)
comsol_time = comsol[0].values
comsol_tpts = len(comsol_time)
comsol_voltage = comsol[1].values
comsol = pd.read_csv("comsol/c_e_C{}.csv".format(C_rate), sep=",", header=None)

comsol_c_e_npts = int(len(comsol[0].values) / comsol_tpts)
comsol_c_e_pts = comsol[0].values[0:comsol_c_e_npts]
comsol_c_e_vals = np.reshape(
    comsol[1].values, (comsol_c_e_npts, comsol_tpts), order="F"
)

"-----------------------------------------------------------------------------"
"Create and solve pybamm model"

# load model and geometry
model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry

# load parameters and process model and geometry
param = model.default_parameter_values
param["Electrode depth"] = 1
param["Electrode height"] = 1
param["Typical current"] = 24 * C_rate
param.process_model(model)
param.process_geometry(geometry)

# create mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 11, var.x_s: 5, var.x_p: 11, var.r_n: 11, var.r_p: 11}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
solver = model.default_solver
t = np.linspace(0, 1, 500)
solver.solve(model, t)

"-----------------------------------------------------------------------------"
"Get variables for comparison"

tau = pybamm.standard_parameters_lithium_ion.tau_discharge
tau_eval = param.process_symbol(tau).evaluate(0, 0)
time = comsol_time / tau_eval

x_electrolyte = comsol_c_e_pts / (
    param["Negative electrode width"]
    + param["Separator width"]
    + param["Positive electrode width"]
)

discharge_capacity = pybamm.ProcessedVariable(
    model.variables["Discharge capacity [Ah]"], solver.t, solver.y, mesh=mesh
)
discharge_capacity_sol = discharge_capacity(solver.t)
comsol_discharge_capacity = comsol_time * param["Typical current"] / 3600

voltage = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solver.t, solver.y, mesh=mesh
)
voltage_sol = voltage(solver.t)

c_e = pybamm.ProcessedVariable(
    model.variables["Electrolyte concentration [mols m-3]"],
    solver.t,
    solver.y,
    mesh=mesh,
)

"-----------------------------------------------------------------------------"
"Make plots"

# discharge curve
plt.plot(comsol_discharge_capacity, comsol_voltage, "r:", label="Comsol")
plt.plot(discharge_capacity_sol, voltage_sol, "b-", label="PyBaMM")
plt.xlim([0, 26])
plt.ylim([3.2, 3.9])
plt.legend(loc="best")
plt.xlabel("Discharge Capacity (Ah)")
plt.ylabel("Voltage (V)")
plt.tight_layout()
plt.show()


# electrolyte concentration
def plot_electrolyte_concentrations(ind):
    plt.figure(figsize=(15, 8))
    plt.plot(x_electrolyte, c_e(time[ind], x_electrolyte), "b-")
    plt.plot(x_electrolyte, comsol_c_e_vals[:, ind], "r:")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$c_e$ (mol/m$^3$)")
    plt.show()


for ind in range(comsol_tpts):
    plot_electrolyte_concentrations(ind)
