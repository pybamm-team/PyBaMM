"""DC1 and DFN on an LG M50 cell, with the parameters of Paten et al. Table IV."""

import numpy as np

import pybamm
import pybamm_model_zoo as zoo
from pybamm_model_zoo.dual_continuum import parameter_sets

models = {
    "DFN": (pybamm.lithium_ion.DFN(), parameter_sets.paten2026_dfn()),
    "DC1": (zoo.load("DualContinuum")(), parameter_sets.paten2026_dc1()),
}

times = np.linspace(0, 3400, 69)
voltages = {}
for label, (model, parameter_values) in models.items():
    solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
        [0, 3600], t_interp=times
    )
    voltages[label] = solution["Voltage [V]"](times)
    print(
        f"{label}: discharge ends at {solution.t[-1]:.0f} s, solve {solution.solve_time}"
    )

print("  time [s]   V_DFN [V]   V_DC1 [V]")
for t, v_dfn, v_dc1 in zip(
    times[::10], voltages["DFN"][::10], voltages["DC1"][::10], strict=True
):
    print(f"{t:10.0f}   {v_dfn:9.4f}   {v_dc1:9.4f}")
