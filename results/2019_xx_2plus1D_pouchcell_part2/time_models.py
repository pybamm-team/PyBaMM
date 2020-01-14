import pybamm
import numpy as np

import pickle

import models


path = "/home/scott/Projects/PyBaMM/results/2019_xx_2plus1D_pouchcell_part2/"

pybamm.set_logging_level("INFO")
param = {}
thermal = False
t_eval = np.linspace(0, 0.16, 100)

var_pts = {
    pybamm.standard_spatial_vars.x_n: 5,
    pybamm.standard_spatial_vars.x_s: 5,
    pybamm.standard_spatial_vars.x_p: 5,
    pybamm.standard_spatial_vars.r_n: 5,
    pybamm.standard_spatial_vars.r_p: 5,
    pybamm.standard_spatial_vars.y: 5,
    pybamm.standard_spatial_vars.z: 5,
}

models = {
    "2+1D DFN": models.DFN_2p1D(thermal, param),
    "2+1D SPM": models.SPM_2p1D(thermal, param),
    "2+1D SPMe": models.SPMe_2p1D(thermal, param),
    # "DFNCC": models.DFNCC(thermal, param),
    "1D DFN": models.DFN(thermal, param),
    "SPMeCC": models.SPMe(thermal, param),
    "SPM": models.SPM(thermal, param),
}

solvers = {
    "2+1D DFN": pybamm.CasadiSolver(mode="fast"),
    "2+1D SPM": pybamm.CasadiSolver(mode="fast"),
    "2+1D SPMe": pybamm.CasadiSolver(mode="fast"),
    "DFNCC": pybamm.CasadiSolver(mode="fast"),
    "1D DFN": pybamm.CasadiSolver(mode="fast"),
    "SPMeCC": pybamm.CasadiSolver(mode="fast"),
    "SPM": pybamm.CasadiSolver(mode="fast"),
}

c_rate = 1

setup_times = {}
solve_times = {}

for model_name, model in models.items():
    model.solve(var_pts, c_rate, t_eval, solvers[model_name])
    setup_times[model_name] = model.sim.solution.set_up_time
    solve_times[model_name] = model.sim.solution.solve_time

print(setup_times)
print(solve_times)
pickle.dump(setup_times, open(path + "setup_times.p", "wb"))
pickle.dump(solve_times, open(path + "solve_times.p", "wb"))
