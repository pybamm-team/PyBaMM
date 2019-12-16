import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pickle

import models

path = "/home/scott/Projects/PyBaMM/results/2019_xx_2plus1D_pouchcell_part2/"
pybamm.set_logging_level("INFO")

load = True
thermal = True
c_rate = 1
t_eval = np.linspace(0, 0.17, 100)

var_pts = {
    pybamm.standard_spatial_vars.x_n: 5,
    pybamm.standard_spatial_vars.x_s: 5,
    pybamm.standard_spatial_vars.x_p: 5,
    pybamm.standard_spatial_vars.r_n: 5,
    pybamm.standard_spatial_vars.r_p: 5,
    pybamm.standard_spatial_vars.y: 5,
    pybamm.standard_spatial_vars.z: 5,
}

param = {
    # "Heat transfer coefficient [W.m-2.K-1]": 0.1,
    # "Negative current collector conductivity [S.m-1]": 5.96e6,
    # "Positive current collector conductivity [S.m-1]": 3.55e6,
    "Negative current collector conductivity [S.m-1]": 5.96e6,
    "Positive current collector conductivity [S.m-1]": 3.55e6,
}

if load is False:
    models = {
        "2+1D DFN": models.DFN_2p1D(thermal, param),
        "2+1D SPM": models.SPM_2p1D(thermal, param),
        "2+1D SPMe": models.SPMe_2p1D(thermal, param),
        "1D DFN": models.DFN(thermal, param),
        "DFNCC": models.DFNCC(thermal, param),
        "SPM": models.SPM(thermal, param),
        "SPMeCC": models.SPMeCC(thermal, param),
    }

linestyles = {
    "2+1D DFN": "-",
    "2+1D SPM": ":",
    "2+1D SPMe": "--",
    "1D DFN": ":",
    "DFNCC": "--",
    "SPM": ":",
    "SPMeCC": "--",
}

colors = [(0.89, 0.1, 0.1), (0.21, 0.49, 0.72), (0.3, 0.68, 0.68), (0.59, 0.3, 0.64)]
colors = {
    "2+1D DFN": "k",
    "2+1D SPM": (0.21, 0.49, 0.72),
    "2+1D SPMe": (0.21, 0.49, 0.72),
    "1D DFN": (0.32, 0.64, 0.11),
    "DFNCC": (0.32, 0.64, 0.11),
    "SPM": (0.89, 0.1, 0.1),
    "SPMeCC": (0.89, 0.1, 0.1),
}

markers = {
    "2+1D DFN": None,
    "2+1D SPM": None,
    "2+1D SPMe": None,
    "1D DFN": "o",
    "DFNCC": "s",
    "SPM": "^",
    "SPMeCC": "D",
}


if load is False:
    voltage_solutions = {}

    for model_name, model in models.items():
        model.solve(var_pts, c_rate, t_eval)

        variables = ["Time [h]", "Discharge capacity [A.h]", "Terminal voltage [V]"]
        pv = model.processed_variables(variables)

        time_hours = pv["Time [h]"](model.t)
        dc = pv["Discharge capacity [A.h]"](model.t)
        tv = pv["Terminal voltage [V]"](model.t)
        voltage_solutions[model_name] = (time_hours, dc, tv)

    pickle.dump(voltage_solutions, open(path + "voltage_solutions.p", "wb"))

else:
    voltage_solutions = pickle.load(open(path + "voltage_solutions.p", "rb"))

lines = {}
for model_name, solutions in voltage_solutions.items():

    time, dc, voltage = solutions

    (lines[model_name],) = plt.plot(
        dc,
        voltage,
        label=model_name,
        linestyle=linestyles[model_name],
        color=colors[model_name],
        marker=None,
        markersize=5,
    )

plt.legend()

plt.xlim([0, 0.65])
plt.ylim([3.5, 3.8])
plt.xlabel("Discharge capacity [A.h]")
plt.ylabel("Terminal voltage [V]")
plt.show()
