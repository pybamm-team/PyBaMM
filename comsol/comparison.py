import pybamm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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
param["Electrode depth [m]"] = 1
param["Electrode height [m]"] = 1
param["Typical current [A]"] = 24 * C_rate
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

x = np.linspace(0, 1, 100)
comsol_x_electrolyte = comsol_c_e_pts / comsol_c_e_pts[-1]

discharge_capacity = pybamm.ProcessedVariable(
    model.variables["Discharge capacity [A.h]"], solver.t, solver.y, mesh=mesh
)
discharge_capacity_sol = discharge_capacity(solver.t)
comsol_discharge_capacity = comsol_time * param["Typical current [A]"] / 3600

voltage = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solver.t, solver.y, mesh=mesh
)
voltage_sol = voltage(solver.t)

c_e = pybamm.ProcessedVariable(
    model.variables["Electrolyte concentration [mol.m-3]"],
    solver.t,
    solver.y,
    mesh=mesh,
)

"-----------------------------------------------------------------------------"
"Make plots"

fig, ax = plt.subplots(figsize=(15, 8))
plt.tight_layout()
plt.subplots_adjust(left=-0.1)

# discharge curve
v_min = 3.2
v_max = 3.9
plt.subplot(121)
time_tracer, = plt.plot(
    [comsol_discharge_capacity[0], comsol_discharge_capacity[0]], [v_min, v_max], "k--"
)
plt.plot(comsol_discharge_capacity, comsol_voltage, "r:", label="Comsol")
plt.plot(discharge_capacity_sol, voltage_sol, "b-", label="PyBaMM")
plt.axis([0, 26, v_min, v_max])
plt.xlabel("Discharge Capacity (Ah)")
plt.ylabel("Voltage (V)")
plt.legend(loc="best")

# electrolyte concentration
plt.subplot(122)
c_e_plot, = plt.plot(x, c_e(time[0], x), "b-")
comsol_c_e_plot, = plt.plot(comsol_x_electrolyte, comsol_c_e_vals[:, 0], "r:")
plt.axis([0, 1, 700, 1300])
plt.xlabel(r"$x$")
plt.ylabel(r"$c_e$ (mol/m$^3$)")

axcolor = "lightgoldenrodyellow"
axfreq = plt.axes([0.315, 0.02, 0.37, 0.03], facecolor=axcolor)
sfreq = Slider(axfreq, "Time", 0, comsol_time[-1], valinit=0)


def update_plot(t):
    # find t index
    ind = (np.abs(comsol_time - t)).argmin()
    # update time
    time_tracer.set_xdata(comsol_discharge_capacity[ind])
    # update electrolyte concentration
    c_e_plot.set_ydata(c_e(time[ind], x))
    comsol_c_e_plot.set_ydata(comsol_c_e_vals[:, ind])
    fig.canvas.draw_idle()


sfreq.on_changed(update_plot)
plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5)
plt.show()
