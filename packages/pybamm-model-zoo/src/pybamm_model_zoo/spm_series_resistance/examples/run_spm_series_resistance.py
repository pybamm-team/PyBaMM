"""Compare the SPM with and without a lumped series resistance."""

import numpy as np

import pybamm
import pybamm_model_zoo as zoo


def solve(series_resistance):
    model = zoo.load("SPMSeriesResistance")()
    parameter_values = model.default_parameter_values
    parameter_values["Series resistance [Ohm]"] = series_resistance
    simulation = pybamm.Simulation(model, parameter_values=parameter_values)
    return simulation.solve([0, 1800])


solutions = {resistance: solve(resistance) for resistance in (0.0, 0.05)}
times = np.linspace(0, 1800, 7)

print("  t [s]   V(R=0) [V]   V(R=0.05) [V]   drop [V]")
for time in times:
    without = solutions[0.0]["Voltage [V]"](time)
    with_resistance = solutions[0.05]["Voltage [V]"](time)
    print(
        f"{time:7.0f}   {without:10.5f}   {with_resistance:13.5f}   "
        f"{without - with_resistance:8.5f}"
    )
