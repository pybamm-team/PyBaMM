import pybamm
import os
import numpy as np
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()
os.chdir(pybamm.__path__[0] + "/..")


model = pybamm.lithium_ion.BasicDFNComposite()
chemistry = pybamm.parameter_sets.Chen2020_composite
param = pybamm.ParameterValues(chemistry=chemistry)
# model = pybamm.lithium_ion.DFN()
# param = model.default_parameter_values
param.update({"Upper voltage cut-off [V]": 4.5})
param.update({"Lower voltage cut-off [V]": 2})

C_rates = 0.1
t_evals = np.linspace(0, 7200 / C_rates, 200)
capacity = param["Nominal cell capacity [A.h]"]
current = C_rates * capacity


def I_fun(A):
    def current(t):
        period = 3600 / 0.1
        discharge = pybamm.sigmoid( t - period, 0, 1)
        charge = pybamm.sigmoid(- t + period, 0, 1)
        return A * discharge - A * charge
        # return A * (t < period) - A * (t >= period)
        # return A * 0.2 * pybamm.sin(2 * np.pi * t / 7200 )
    return current


param["Current function [A]"] = I_fun(current)

sim1 = pybamm.Simulation(
    model,
    parameter_values=param,
    solver=pybamm.CasadiSolver(),
)
solution = sim1.solve(t_eval=t_evals)
# plot
plot = pybamm.QuickPlot( solution, [       
        "Current [A]",
        "Terminal voltage [V]",
        "X-averaged negative electrode open circuit potential of phase 1 [V]",
        "X-averaged negative electrode open circuit potential of phase 2 [V]",
    ]
)
plot.dynamic_plot()
stop = timeit.default_timer()
print("running time: " + str(stop - start) + "s")


# %%
