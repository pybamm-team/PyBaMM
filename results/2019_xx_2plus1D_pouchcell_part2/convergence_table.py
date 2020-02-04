import pybamm
import numpy as np
import matplotlib.pyplot as plt

import models

import pickle

# path = "/home/scott/Projects/PyBaMM/results/2019_xx_2plus1D_pouchcell_part2/"
pybamm.set_logging_level("INFO")


load = False
thermal = True
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


def rms_error(a, b):
    return np.sqrt(sum((a - b) ** 2) / a.size)


def mean_abs_error(a, b):
    return np.mean(np.abs(voltage - truth_voltage))


if load is False:
    abs_errors = {}
    rms_errors = {}
    for model_name in models.keys():
        if model_name != truth:
            abs_errors[model_name] = {}
            rms_errors[model_name] = {}

    solved_models = {}
    final_times = {}

    # get the solutions
    solved_models = {}
    final_time = 100

    for model_name, model in models.items():
        model.solve(var_pts, 1, t_eval)
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
            abs_errors[model_name] = mean_abs_error(voltage, truth_voltage)
            rms_errors[model_name] = rms_error(voltage, truth_voltage)
