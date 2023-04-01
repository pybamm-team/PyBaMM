import pybamm
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# dictionary of available comsol results
C_rates = {"01": 0.1, "05": 0.5, "1": 1, "2": 2, "3": 3}

# load model and geometry
model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry


# load parameters and process model and geometry
param = model.default_parameter_values
param.update(
    {
        "Electrode width [m]": 1,
        "Electrode height [m]": 1,
        "Negative electrode conductivity [S.m-1]": 126,
        "Positive electrode conductivity [S.m-1]": 16.6,
        "Current function [A]": "[input]",
    }
)
param.process_model(model)
param.process_geometry(geometry)

# create mesh
var_pts = {"x_n": 31, "x_s": 11, "x_p": 31, "r_n": 11, "r_p": 11}
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
plt.xlabel(r"Discharge Capacity (Ah)", fontsize=20)
plt.ylabel("Voltage (V)", fontsize=20)
plt.title(r"Comsol $\cdots$ PyBaMM $-$", fontsize=20)
voltage_difference_plot = plt.subplot(212)
plt.xlim([0, 26])
plt.yscale("log")
plt.grid(True)
plt.xlabel(r"Discharge Capacity (Ah)", fontsize=20)
plt.ylabel(r"$\vert V - V_{comsol} \vert$", fontsize=20)

for key, C_rate in C_rates.items():
    current = 24 * C_rate
    # load the comsol results
    comsol_results_path = pybamm.get_parameters_filepath(
        "input/comsol_results/comsol_{}C.pickle".format(key)
    )
    comsol_variables = pickle.load(open(comsol_results_path, "rb"))
    comsol_time = comsol_variables["time"]
    comsol_voltage = comsol_variables["voltage"]

    # solve model at comsol times
    t = comsol_time
    solution = pybamm.CasadiSolver(mode="fast").solve(
        model, t, inputs={"Current function [A]": current}
    )
    time_in_seconds = solution["Time [s]"].entries

    # discharge capacity
    discharge_capacity = solution["Discharge capacity [A.h]"]
    discharge_capacity_sol = discharge_capacity(time_in_seconds)
    comsol_discharge_capacity = comsol_time * current / 3600

    # extract the voltage
    voltage = solution["Voltage [V]"]
    voltage_sol = voltage(time_in_seconds)

    # calculate the difference between the two solution methods
    end_index = min(len(time_in_seconds), len(comsol_time))
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
