import pybamm
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# increase recursion limit for large expression trees
sys.setrecursionlimit(100000)

pybamm.set_logging_level("INFO")


"-----------------------------------------------------------------------------"
"Run PyBamm Models"

# models
options = {"current collector": "potential pair", "dimensionality": 1}
models = [
    pybamm.lithium_ion.DFN(name="PyBaMM 1D"),
    pybamm.lithium_ion.DFN(options, name="PyBaMM 1+1D"),
]

# parameters
param = models[0].default_parameter_values
param.update({"C-rate": 1})

# set npts
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 5,
    var.x_s: 5,
    var.x_p: 5,
    var.r_n: 21,
    var.r_p: 21,
    var.z: 10,
}

# solver
solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6, mode="fast")
tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_end = 3555 / tau
t_eval = np.linspace(0, t_end, 100)

# build and run simulations
simulations = [None] * len(models)
for i, model in enumerate(models):
    simulations[i] = pybamm.Simulation(
        models[i], parameter_values=param, var_pts=var_pts, solver=solver,
    )
    simulations[i].build(check_model=False)
    simulations[i].solve(t_eval=t_eval)

    # plot discharge curve
    output_variables = simulations[i].post_process_variables("Terminal voltage [V]")
    pybamm_v = output_variables["Terminal voltage [V]"](t_eval)
    plt.plot(t_eval * tau, pybamm_v, "-", label=model.name)


"-----------------------------------------------------------------------------"
"Plot comsol discharge curves"

comsol_variables = pickle.load(
    open("input/comsol_results/comsol_isothermal_1C.pickle", "rb")
)
t_plot = comsol_variables["time"]  # dimensional in seconds
comsol_v = comsol_variables["voltage"]
plt.plot(t_plot, comsol_v, "o", fillstyle="none", label="COMSOL 1D")

comsol_variables = pickle.load(
    open("input/comsol_results/comsol_isothermal_1plus1D_1C.pickle", "rb")
)
t_plot = comsol_variables["time"]  # dimensional in seconds
comsol_v = comsol_variables["voltage"]
plt.plot(t_plot, comsol_v, "o", fillstyle="none", label="COMSOL 1+1D")

comsol_variables = pickle.load(
    open("input/comsol_results/comsol_isothermal_2plus1D_1C.pickle", "rb")
)
t_plot = comsol_variables["time"]  # dimensional in seconds
comsol_v = comsol_variables["voltage"]
plt.plot(t_plot, comsol_v, "o", fillstyle="none", label="COMSOL 2+1D")

plt.xlabel("t [s]")
plt.ylabel("Terminal voltage [V]")
plt.legend()
plt.show()
