import pybamm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

"-----------------------------------------------------------------------------"
"Pick C_rate and load comsol data"

# C_rate
C_rate = 1

# time-voltage
comsol = pd.read_csv("comsol/Voltage_C{}.csv".format(C_rate), sep=",", header=None)
comsol_time = comsol[0].values
comsol_time_npts = len(comsol_time)
comsol_voltage = comsol[1].values

# electrolyte concentration
comsol = pd.read_csv("comsol/c_e_C{}.csv".format(C_rate), sep=",", header=None)
comsol_x_npts = int(len(comsol[0].values) / comsol_time_npts)
comsol_x = comsol[0].values[0:comsol_x_npts]
comsol_c_e_vals = np.reshape(
    comsol[1].values, (comsol_x_npts, comsol_time_npts), order="F"
)

# electrolyte potential
comsol = pd.read_csv("comsol/phi_e_C{}.csv".format(C_rate), sep=",", header=None)
comsol_phi_e_vals = np.reshape(
    comsol[1].values, (comsol_x_npts, comsol_time_npts), order="F"
)

# negative electrode potential

# positive electrode potential


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
# Same number of points as default lionsimba
var_pts = {var.x_n: 31, var.x_s: 31, var.x_p: 31, var.r_n: 11, var.r_p: 11}
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

# discharge timescale
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge).evaluate(0, 0)
time = comsol_time / tau

# discharge capacity
discharge_capacity = pybamm.ProcessedVariable(
    model.variables["Discharge capacity [A.h]"], solver.t, solver.y, mesh=mesh
)
discharge_capacity_sol = discharge_capacity(solver.t)
comsol_discharge_capacity = comsol_time * param["Typical current [A]"] / 3600

# spatial points
l_n = param.process_symbol(pybamm.geometric_parameters.l_n).evaluate(0, 0)
l_s = param.process_symbol(pybamm.geometric_parameters.l_s).evaluate(0, 0)
l_p = param.process_symbol(pybamm.geometric_parameters.l_p).evaluate(0, 0)
x_n = np.linspace(0, l_n, 40)
x_s = np.linspace(l_n, l_n + l_s, 20)
x_p = np.linspace(l_n + l_s, 1, 40)
x = np.linspace(0, 1, 100)

# voltage
voltage = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solver.t, solver.y, mesh=mesh
)
voltage_sol = voltage(solver.t)

# electrolyte concentration
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
comsol_c_e_plot, = plt.plot(comsol_x / comsol_x[-1], comsol_c_e_vals[:, 0], "r:")
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
