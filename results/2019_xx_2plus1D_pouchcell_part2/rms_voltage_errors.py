import pybamm
import numpy as np
import matplotlib.pyplot as plt

import models

import pickle

path = "/home/scott/Projects/PyBaMM/results/2019_xx_2plus1D_pouchcell_part2/"
pybamm.set_logging_level("INFO")


load = True
thermal = True
c_rates = {"0.1 C": 0.1, "1 C": 1, "2 C": 2, "3 C": 3, "5 C": 5}
# c_rates = {"1 C": 1}
t_eval = np.linspace(0, 0.17, 100)

truth = "2+1D DFN"
time_pts = 100

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


def rms_error(a, b):
    return np.sqrt(sum((a - b) ** 2) / a.size)


def mean_abs_error(a, b):
    return np.mean(np.abs(voltage - truth_voltage))


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
    "2+1D SPM": "o",
    "2+1D SPMe": "o",
    "1D DFN": "s",
    "DFNCC": "s",
    "SPM": "D",
    "SPMeCC": "D",
}

# construct error arrays

if load is False:
    abs_errors = {}
    rms_errors = {}
    number_of_c_rates = len(c_rates)
    for model_name in models.keys():
        if model_name != truth:
            abs_errors[model_name] = {}
            rms_errors[model_name] = {}

    solved_models = {}
    final_times = {}

    # get the solutions
    for c_rate_name, c_rate in c_rates.items():

        solved_models = {}
        final_time = 100

        for model_name, model in models.items():
            model.solve(var_pts, c_rate, t_eval)
            solved_models[model_name] = model
            final_time = min(model.t[-1], final_time)

        t = np.linspace(0, final_time, time_pts)

        # extract truth
        truth_model = solved_models[truth]
        truth_variables = truth_model.processed_variables(["Terminal voltage [V]"])
        truth_voltage = truth_variables["Terminal voltage [V]"](t)

        for model_name, model in models.items():
            if model_name != truth:
                variables = model.processed_variables(["Terminal voltage [V]"])
                voltage = variables["Terminal voltage [V]"](t)
                abs_errors[model_name][c_rate_name] = mean_abs_error(
                    voltage, truth_voltage
                )
                rms_errors[model_name][c_rate_name] = rms_error(voltage, truth_voltage)

    pickle.dump(rms_errors, open(path + "rms_errors.p", "wb"))

else:
    rms_errors = pickle.load(open(path + "rms_errors.p", "rb"))


lines = {}
for model_name, error_dict in rms_errors.items():

    if model_name != truth:
        c_rate_array = np.array(list(c_rates.values()))
        error_array = np.array(list(error_dict.values()))

        lines[model_name] = plt.plot(
            c_rate_array,
            error_array,
            linestyle=linestyles[model_name],
            color=colors[model_name],
            marker=markers[model_name],
            markersize=5,
            label=model_name,
        )


# add legends
plt.legend()

# plt.xlim([0, 0.65])
# plt.ylim([3.5, 3.8])
plt.xlabel("C-rate")
plt.ylabel("RMS Voltage Error [V]")
plt.show()
