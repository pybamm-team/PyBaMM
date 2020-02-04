import pybamm
import numpy as np

import models
import pickle

import sys

# path = "/home/scott/Projects/PyBaMM/results/2019_xx_2plus1D_pouchcell_part2/"
pybamm.set_logging_level("INFO")


load = False
thermal = True
t_eval = np.linspace(0, 0.14, 100)

truth = "2+1D DFN"
time_pts = 100

pts = [3, 5]

if load is True:
    abs_errors = pickle.load(open("abs_errors.p", "rb"))
    rms_errors = pickle.load(open("rms_errors.p", "rb"))
    print(abs_errors)
    print(rms_errors)
    sys.exit()


def set_var_pts(num):
    var_pts = {
        pybamm.standard_spatial_vars.x_n: num,
        pybamm.standard_spatial_vars.x_s: num,
        pybamm.standard_spatial_vars.x_p: num,
        pybamm.standard_spatial_vars.r_n: num,
        pybamm.standard_spatial_vars.r_p: num,
        pybamm.standard_spatial_vars.y: num,
        pybamm.standard_spatial_vars.z: num,
    }
    return var_pts


solvers = {
    "2+1D DFN": pybamm.CasadiSolver(mode="fast"),
    "2+1D SPM": pybamm.CasadiSolver(mode="fast"),
    "2+1D SPMe": pybamm.CasadiSolver(mode="fast"),
    "DFNCC": pybamm.CasadiSolver(mode="fast"),
    "1D DFN": pybamm.CasadiSolver(mode="fast"),
    "SPMeCC": pybamm.CasadiSolver(mode="fast"),
    "SPM": pybamm.CasadiSolver(mode="fast"),
}

param = {
    # "Heat transfer coefficient [W.m-2.K-1]": 0.1,
    # "Negative current collector conductivity [S.m-1]": 5.96e6,
    # "Positive current collector conductivity [S.m-1]": 3.55e6,
    "Negative current collector conductivity [S.m-1]": 5.96e6,
    "Positive current collector conductivity [S.m-1]": 3.55e6,
}

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


# define errors dictionaries
abs_errors = {}
rms_errors = {}
for model_name in models.keys():
    abs_errors[model_name] = {}
    rms_errors[model_name] = {}

# solve all models for all numbers of points
solved_models = {}
final_time = 100  # just some big number
for model_name in models.keys():
    solved_models[model_name] = {}
for pt in pts:
    var_pts = set_var_pts(pt)
    for model_name, model in models.items():
        model.solve(var_pts, 1, t_eval, solvers[model_name])
        solved_models[model_name][str(pt) + " points"] = model
        final_time = min(model.t[-1], final_time)

t = np.linspace(0, final_time, time_pts)

# find truth
truth_pts = max(pts)
truth_model = solved_models["2+1D DFN"][str(truth_pts) + " points"]
truth_variables = truth_model.processed_variables(["Terminal voltage [V]"])
truth_voltage = truth_variables["Terminal voltage [V]"](t)

for pt in pts:
    for model_name, model in models.items():
        variables = model.processed_variables(["Terminal voltage [V]"])
        voltage = variables["Terminal voltage [V]"](t)
        abs_errors[model_name][str(pt) + " points"] = mean_abs_error(
            voltage, truth_voltage
        )
        rms_errors[model_name][str(pt) + " points"] = rms_error(voltage, truth_voltage)

print(abs_errors)
print(rms_errors)
pickle.dump(abs_errors, open("abs_errors.p", "wb"))
pickle.dump(rms_errors, open("rms_errors.p", "wb"))
