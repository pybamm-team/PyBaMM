"""DC1 and DFN on an LG M50 cell, with the parameters of Paten et al. Table IV."""

import matplotlib.pyplot as plt
import numpy as np

import pybamm
import pybamm_model_zoo as zoo

models = {
    "DFN": (pybamm.lithium_ion.DFN(), pybamm.ParameterValues("Paten2026_DFN")),
    "DC1": (zoo.load("DualContinuum")(), pybamm.ParameterValues("Paten2026_DC")),
}

fig, ax = plt.subplots()
for label, (model, parameter_values) in models.items():
    solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
        [0, 3600]
    )
    print(
        f"{label}: discharge ends at {solution.t[-1]:.0f} s, solve {solution.solve_time}"
    )
    times = np.linspace(0, solution.t[-1], 300)
    ax.plot(times / 60, solution["Voltage [V]"](times), label=label)

ax.set_xlabel("Time [min]")
ax.set_ylabel("Voltage [V]")
ax.set_title("LG M50, 1C discharge")
ax.legend()
plt.show()
