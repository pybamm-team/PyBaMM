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
    temperature_solution = {}

    for model_name, model in models.items():
        model.solve(var_pts, c_rate, t_eval)

        variables = [
            "Time [h]",
            "Discharge capacity [A.h]",
            "Volume-averaged cell temperature [K]",
            "Volume-averaged Ohmic heating [W.m-3]",
            "Volume-averaged irreversible electrochemical heating [W.m-3]",
            "Volume-averaged reversible heating [W.m-3]",
        ]
        pv = model.processed_variables(variables)

        time_hours = pv["Time [h]"](model.t)
        dc = pv["Discharge capacity [A.h]"](model.t)
        temperature = pv["Volume-averaged cell temperature [K]"](model.t)

        ohmic_heating = pv["Volume-averaged Ohmic heating [W.m-3]"](model.t)
        irr_heating = pv[
            "Volume-averaged irreversible electrochemical heating [W.m-3]"
        ](model.t)
        rev_heating = pv["Volume-averaged reversible heating [W.m-3]"](model.t)
        temperature_solution[model_name] = (
            time_hours,
            dc,
            temperature,
            ohmic_heating,
            irr_heating,
            rev_heating,
        )

    pickle.dump(temperature_solution, open(path + "vol_av_temperature.p", "wb"))

else:
    temperature_solution = pickle.load(open(path + "vol_av_temperature.p", "rb"))

fig, axes = plt.subplots(1, 5)
lines = {}

for model_name, solutions in temperature_solution.items():

    time, dc, temperature, ohmic_heating, irr_heating, rev_heating = solutions

    (lines[model_name],) = axes[0].plot(
        dc,
        temperature,
        label=model_name,
        linestyle=linestyles[model_name],
        color=colors[model_name],
        marker=None,
        markersize=5,
    )

axes[0].set_xlabel("Discharge capacity [A.h]")
axes[0].set_ylabel("Volume-Averaged cell temperature [K]")

lines = {}
for model_name, solutions in temperature_solution.items():

    time, dc, temperature, ohmic_heating, irr_heating, rev_heating = solutions

    (lines[model_name],) = axes[1].plot(
        dc,
        ohmic_heating + irr_heating + rev_heating,
        label=model_name,
        linestyle=linestyles[model_name],
        color=colors[model_name],
        marker=None,
        markersize=5,
    )

axes[1].set_xlabel("Discharge capacity [A.h]")
axes[1].set_ylabel("Total heating [W.m-3]")


lines = {}
for model_name, solutions in temperature_solution.items():

    time, dc, temperature, ohmic_heating, irr_heating, rev_heating = solutions

    (lines[model_name],) = axes[2].plot(
        dc,
        ohmic_heating,
        label=model_name,
        linestyle=linestyles[model_name],
        color=colors[model_name],
        marker=None,
        markersize=5,
    )

axes[2].set_xlabel("Discharge capacity [A.h]")
axes[2].set_ylabel("Ohmic Heating")

lines = {}
for model_name, solutions in temperature_solution.items():

    time, dc, temperature, ohmic_heating, irr_heating, rev_heating = solutions

    (lines[model_name],) = axes[3].plot(
        dc,
        irr_heating,
        label=model_name,
        linestyle=linestyles[model_name],
        color=colors[model_name],
        marker=None,
        markersize=5,
    )

axes[3].set_xlabel("Discharge capacity [A.h]")
axes[3].set_ylabel("Irreversible heating")

lines = {}
for model_name, solutions in temperature_solution.items():

    time, dc, temperature, ohmic_heating, irr_heating, rev_heating = solutions

    (lines[model_name],) = axes[4].plot(
        dc,
        rev_heating,
        label=model_name,
        linestyle=linestyles[model_name],
        color=colors[model_name],
        marker=None,
        markersize=5,
    )

plt.legend()

axes[4].set_xlabel("Discharge capacity [A.h]")
axes[4].set_ylabel("Reversible heating")

plt.subplots_adjust(left=0.08, bottom=0.17, right=0.96, top=0.9, wspace=0.5, hspace=0.4)

fig.set_figheight(4)
fig.set_figwidth(14)

plt.show()

