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
comsol = pd.read_csv("comsol/data/Voltage_C{}.csv".format(C_rate), sep=",", header=None)
comsol_time = comsol[0].values
comsol_time_npts = len(comsol_time)
comsol_voltage = comsol[1].values

# negative electrode potential
comsol = pd.read_csv("comsol/data/phi_n_C{}.csv".format(C_rate), sep=",", header=None)
comsol_x_n_npts = int(len(comsol[0].values) / comsol_time_npts)
comsol_x_n = comsol[0].values[0:comsol_x_n_npts]
comsol_phi_n_vals = np.reshape(
    comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
)

# negative particle surface concentration
comsol = pd.read_csv(
    "comsol/data/c_n_surf_C{}.csv".format(C_rate), sep=",", header=None
)
comsol_c_n_surf_vals = np.reshape(
    comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
)

# positive electrode potential
comsol = pd.read_csv("comsol/data/phi_p_C{}.csv".format(C_rate), sep=",", header=None)
comsol_x_p_npts = int(len(comsol[0].values) / comsol_time_npts)
comsol_x_p = comsol[0].values[0:comsol_x_p_npts]
comsol_phi_p_vals = np.reshape(
    comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
)

# positive particle surface concentration
comsol = pd.read_csv(
    "comsol/data/c_p_surf_C{}.csv".format(C_rate), sep=",", header=None
)
comsol_c_p_surf_vals = np.reshape(
    comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
)

# electrolyte concentration
comsol = pd.read_csv("comsol/data/c_e_C{}.csv".format(C_rate), sep=",", header=None)
comsol_x_npts = int(len(comsol[0].values) / comsol_time_npts)
comsol_x = comsol[0].values[0:comsol_x_npts]
comsol_c_e_vals = np.reshape(
    comsol[1].values, (comsol_x_npts, comsol_time_npts), order="F"
)

# electrolyte potential
comsol = pd.read_csv("comsol/data/phi_e_C{}.csv".format(C_rate), sep=",", header=None)
comsol_phi_e_vals = np.reshape(
    comsol[1].values, (comsol_x_npts, comsol_time_npts), order="F"
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
var_pts = {var.x_n: 31, var.x_s: 11, var.x_p: 31, var.r_n: 11, var.r_p: 11}
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
tau = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.tau_discharge
).evaluate(0, 0)
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

# negative electrode potential
phi_n = pybamm.ProcessedVariable(
    model.variables["Negative electrode potential [V]"], solver.t, solver.y, mesh=mesh
)

# negative particle surface concentration
c_n_surf = pybamm.ProcessedVariable(
    model.variables["Negative particle surface concentration [mol.m-3]"],
    solver.t,
    solver.y,
    mesh=mesh,
)

# positive electrode potential
phi_p = pybamm.ProcessedVariable(
    model.variables["Positive electrode potential [V]"], solver.t, solver.y, mesh=mesh
)

# positive particle surface concentration
c_p_surf = pybamm.ProcessedVariable(
    model.variables["Positive particle surface concentration [mol.m-3]"],
    solver.t,
    solver.y,
    mesh=mesh,
)

# electrolyte potential
phi_e = pybamm.ProcessedVariable(
    model.variables["Electrolyte potential [V]"], solver.t, solver.y, mesh=mesh
)

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


# negative particle surface concentration
c_n_surf_min = 0.9 * np.min(comsol_c_n_surf_vals)
c_n_surf_max = 1.1 * np.max(comsol_c_n_surf_vals)
plt.subplot(241)
c_n_surf_plot, = plt.plot(x_n, c_n_surf(time[0], x_n), "b-")
comsol_c_n_surf_plot, = plt.plot(
    comsol_x_n / comsol_x[-1], comsol_c_n_surf_vals[:, 0], "r:"
)
plt.axis([0, l_n, c_n_surf_min, c_n_surf_max])
plt.xlabel(r"$x$")
plt.ylabel(r"Surface $c_n$ (mol/m$^3$)")

# electrolyte concentration
c_e_min = 0.9 * np.min(comsol_c_e_vals)
c_e_max = 1.1 * np.max(comsol_c_e_vals)
plt.subplot(242)
c_e_plot, = plt.plot(x, c_e(time[0], x), "b-")
comsol_c_e_plot, = plt.plot(comsol_x / comsol_x[-1], comsol_c_e_vals[:, 0], "r:")
plt.axis([0, 1, c_e_min, c_e_max])
plt.xlabel(r"$x$")
plt.ylabel(r"$c_e$ (mol/m$^3$)")

# negative particle surface concentration
c_p_surf_min = 0.9 * np.min(comsol_c_p_surf_vals)
c_p_surf_max = 1.1 * np.max(comsol_c_p_surf_vals)
plt.subplot(243)
c_p_surf_plot, = plt.plot(x_p, c_n_surf(time[0], x_p), "b-")
comsol_c_p_surf_plot, = plt.plot(
    comsol_x_p / comsol_x[-1], comsol_c_p_surf_vals[:, 0], "r:"
)
plt.axis([l_n + l_s, 1, c_p_surf_min, c_p_surf_max])
plt.xlabel(r"$x$")
plt.ylabel(r"Surface $c_p$ (mol/m$^3$)")

# current
I_min = 0.9 * param["Typical current [A]"]
I_max = 1.1 * param["Typical current [A]"]
plt.subplot(244)
time_tracer, = plt.plot([comsol_time[0], comsol_time[0]], [I_min, I_max], "k--")
plt.plot(
    [comsol_time[0], comsol_time[-1]],
    [param["Typical current [A]"], param["Typical current [A]"]],
    "b-",
)
plt.axis([comsol_time[0], comsol_time[-1] * 1.1, I_min, I_max])
plt.xlabel("Time (s)")
plt.ylabel("Applied current (A)")

# negative electrode potential
phi_n_min = 1.1 * np.min(comsol_phi_n_vals)
phi_n_max = 0.9 * np.max(comsol_phi_n_vals)
plt.subplot(245)
phi_n_plot, = plt.plot(x_n, phi_n(time[0], x_n), "b-")
comsol_phi_n_plot, = plt.plot(comsol_x_n / comsol_x[-1], comsol_phi_n_vals[:, 0], "r:")
plt.axis([0, l_n, phi_n_min, phi_n_max])
plt.xlabel(r"$x$")
plt.ylabel(r"$\phi_n$ (V)")

# electrolyte potential
phi_e_min = 1.1 * np.min(comsol_phi_e_vals)
phi_e_max = 0.9 * np.max(comsol_phi_e_vals)
plt.subplot(246)
phi_e_plot, = plt.plot(x, phi_e(time[0], x), "b-")
comsol_phi_e_plot, = plt.plot(comsol_x / comsol_x[-1], comsol_phi_e_vals[:, 0], "r:")
plt.axis([0, 1, phi_e_min, phi_e_max])
plt.xlabel(r"$x$")
plt.ylabel(r"$\phi_e$ (V)")

# positive electrode potential
phi_p_min = 0.9 * np.min(comsol_phi_p_vals)
phi_p_max = 1.1 * np.max(comsol_phi_p_vals)
plt.subplot(247)
phi_p_plot, = plt.plot(x_p, phi_p(time[0], x_p), "b-")
comsol_phi_p_plot, = plt.plot(comsol_x_p / comsol_x[-1], comsol_phi_p_vals[:, 0], "r:")
plt.axis([l_n + l_s, 1, phi_p_min, phi_p_max])
plt.xlabel(r"$x$")
plt.ylabel(r"$\phi_p$ (V)")

# discharge curve
v_min = 3.2
v_max = 3.9
plt.subplot(248)
discharge_capacity_tracer, = plt.plot(
    [comsol_discharge_capacity[0], comsol_discharge_capacity[0]], [v_min, v_max], "k--"
)
plt.plot(comsol_discharge_capacity, comsol_voltage, "r:", label="Comsol")
plt.plot(discharge_capacity_sol, voltage_sol, "b-", label="PyBaMM")
plt.axis(
    [comsol_discharge_capacity[0], comsol_discharge_capacity[-1] * 1.1, v_min, v_max]
)
plt.xlabel("Discharge Capacity (Ah)")
plt.ylabel("Voltage (V)")
plt.legend(loc="best")

axcolor = "lightgoldenrodyellow"
axfreq = plt.axes([0.315, 0.02, 0.37, 0.03], facecolor=axcolor)
sfreq = Slider(axfreq, "Time", 0, comsol_time[-1], valinit=0)


def update_plot(t):
    # find t index
    ind = (np.abs(comsol_time - t)).argmin()
    # update time
    time_tracer.set_xdata(comsol_time[ind])
    discharge_capacity_tracer.set_xdata(comsol_discharge_capacity[ind])
    # update negative particle surface concentration
    c_n_surf_plot.set_ydata(c_n_surf(time[ind], x_n))
    comsol_c_n_surf_plot.set_ydata(comsol_c_n_surf_vals[:, ind])
    # update electrolyte concentration
    c_e_plot.set_ydata(c_e(time[ind], x))
    comsol_c_e_plot.set_ydata(comsol_c_e_vals[:, ind])
    # update negative particle surface concentration
    c_p_surf_plot.set_ydata(c_p_surf(time[ind], x_p))
    comsol_c_p_surf_plot.set_ydata(comsol_c_p_surf_vals[:, ind])
    # update negative electrode potential
    phi_n_plot.set_ydata(phi_n(time[ind], x_n))
    comsol_phi_n_plot.set_ydata(comsol_phi_n_vals[:, ind])
    # update electrolyte potential
    phi_e_plot.set_ydata(phi_e(time[ind], x))
    comsol_phi_e_plot.set_ydata(comsol_phi_e_vals[:, ind])
    # update positive electrode potential
    phi_p_plot.set_ydata(phi_p(time[ind], x_p))
    comsol_phi_p_plot.set_ydata(comsol_phi_p_vals[:, ind])
    fig.canvas.draw_idle()


sfreq.on_changed(update_plot)
plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5)
plt.show()
