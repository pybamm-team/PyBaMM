import pybamm
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# change working directory to the root of pybamm
os.chdir(pybamm.__path__[0] + "/..")

# dictionary of available comsol results
C_rates = {"01": 0.1, "05": 0.5, "1": 1, "2": 2, "3": 3}

# load model and geometry
model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry

# load parameters and process model and geometry
param = model.default_parameter_values
param["Electrode depth [m]"] = 1
param["Electrode height [m]"] = 1
param.process_model(model)
param.process_geometry(geometry)

# create mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 31, var.x_s: 11, var.x_p: 31, var.r_n: 11, var.r_p: 11}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# loop over C_rates dict to create plot
fig, ax = plt.subplots(figsize=(15, 8))
plt.tight_layout()
plt.subplots_adjust(left=-0.1)
discharge_curve = plt.subplot(211)
plt.xlim([0, 26])
plt.ylim([3.2, 3.9])
plt.xlabel(r"Discharge Capacity (Ah)")
plt.ylabel("Voltage (V)")
plt.title(r"Comsol $\cdots$ PyBaMM $-$")
voltage_difference_plot = plt.subplot(212)
plt.xlim([0, 26])
plt.yscale("log")
plt.grid(True)
plt.xlabel(r"Discharge Capacity (Ah)")
plt.ylabel(r"$\vert V - V_{comsol} \vert$")

for key, C_rate in C_rates.items():

    # load the comsol voltage data
    comsol = pd.read_csv(
        "input/comsol_results/{}C/Voltage.csv".format(key), sep=",", header=None
    )
    comsol_time = comsol[0].values
    comsol_voltage = comsol[1].values

    # update current density
    param["Typical current [A]"] = 24 * C_rate
    param.update_model(model, disc)

    # discharge timescale
    tau = param.process_symbol(
        pybamm.standard_parameters_lithium_ion.tau_discharge
    ).evaluate(0, 0)

    # solve model at comsol times
    solver = model.default_solver
    t = comsol_time / tau
    solver.solve(model, t)

    # discharge capacity
    discharge_capacity = pybamm.ProcessedVariable(
        model.variables["Discharge capacity [A.h]"], solver.t, solver.y, mesh=mesh
    )
    discharge_capacity_sol = discharge_capacity(solver.t)
    comsol_discharge_capacity = comsol_time * param["Typical current [A]"] / 3600

    # extract the voltage
    voltage = pybamm.ProcessedVariable(
        model.variables["Terminal voltage [V]"], solver.t, solver.y, mesh=mesh
    )
    voltage_sol = voltage(solver.t)

    # calculate the difference between the two solution methods
    end_index = min(len(solver.t), len(comsol_time))
    voltage_difference = np.abs(voltage_sol[0:end_index] - comsol_voltage[0:end_index])

    # plot discharge curves and absolute voltage_difference
    color = next(ax._get_lines.prop_cycler)["color"]
    discharge_curve.plot(
        comsol_discharge_capacity, comsol_voltage, color=color, linestyle=":"
    )
    discharge_curve.plot(
        discharge_capacity_sol,
        voltage_sol,
        color=color,
        linestyle="-",
        label="{} C".format(C_rate),
    )
    voltage_difference_plot.plot(
        discharge_capacity_sol[0:end_index], voltage_difference, color=color
    )

discharge_curve.legend(loc="best")
plt.subplots_adjust(
    top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35
)
plt.show()
