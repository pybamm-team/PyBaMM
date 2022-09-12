import pybamm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

owd = os.getcwd()
# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

voltage_data_1C = pd.read_csv("pybamm/input/discharge_data/Ecker2015/Ecker_1C.csv", header=None).to_numpy()
voltage_data_5C = pd.read_csv("pybamm/input/discharge_data/Ecker2015/Ecker_5C.csv", header=None).to_numpy()

# choose DFN
model = pybamm.lithium_ion.DFN()

# pick parameters, keeping C-rate as an input to be changed for each solve
parameter_values = pybamm.ParameterValues("Ecker2015")
parameter_values.update({"Current function [A]": "[input]"})

var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: int(parameter_values.evaluate(model.param.n.L / 1e-6)),
    var.x_s: int(parameter_values.evaluate(model.param.s.L / 1e-6)),
    var.x_p: int(parameter_values.evaluate(model.param.p.L / 1e-6)),
    var.r_n: int(parameter_values.evaluate(model.param.n.R_typ / 1e-7)),
    var.r_p: int(parameter_values.evaluate(model.param.p.R_typ / 1e-7)),
}
sim =  pybamm.Simulation(model, parameter_values=parameter_values, var_pts=var_pts)
C_rates = [1, 5]  # C-rates to solve for
capacity = parameter_values["Nominal cell capacity [A.h]"]
t_evals = [
    np.linspace(0, 3800, 100), 
    np.linspace(0, 720, 100)
] # times to return the solution at
solutions = [None] * len(C_rates)  # empty list that will hold solutions

# loop over C-rates
for i, C_rate in enumerate(C_rates):
    current = C_rate * capacity
    sim.solve(t_eval=t_evals[i], solver=pybamm.CasadiSolver(mode="fast"),inputs={"Current function [A]": current})
    solutions[i] = sim.solution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# plot the 1C results
t_sol = solutions[0]["Time [s]"].entries
ax1.plot(t_sol, solutions[0]["Terminal voltage [V]"](t_sol))
ax1.plot(voltage_data_1C[:,0], voltage_data_1C[:,1], "o")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Voltage [V]")
ax1.set_title("1C")
ax1.legend(["DFN", "Experiment"], loc="best")

# plot the 5C results
t_sol = solutions[1]["Time [s]"].entries
ax2.plot(t_sol, solutions[1]["Terminal voltage [V]"](t_sol))
ax2.plot(voltage_data_5C[:,0], voltage_data_5C[:,1], "o")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Voltage [V]")
ax2.set_title("5C")
ax2.legend(["DFN", "Experiment"], loc="best")

plt.tight_layout()
os.chdir(owd)
plt.savefig(f"../benchmarks/benchmark_images/ecker_comparison_{pybamm.__version__}.png")


# content = f"# PyBaMM {pybamm.__version__}\n## Ecker comparison\n<img src='./benchmark_images/ecker_comparison_{pybamm.__version__}.png'>\n"  # noqa

# with open("./benchmarks/validation.md", "r") as original:
#     data = original.read()
# with open("./benchmarks/validation.md", "w") as modified:
#     modified.write(f"{content}\n{data}")