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
param.update({"Negative electrode mass fraction of phase 1": 0.9})
param.update({"Negative electrode mass fraction of phase 2": 0.1})

C_rates = 0.01
period = 3600 / C_rates
rest_period = 3600
transition = 50
t1 = np.linspace(0, rest_period - transition, 201)
t2 = np.linspace(rest_period - transition, rest_period + transition, 51)
t3 = np.linspace(rest_period + transition, rest_period + period - transition, 401)
t4 = np.linspace(rest_period + period - transition, rest_period + period + transition, 51)
t5 = np.linspace(rest_period + period + transition, rest_period + period * 2, 401)
t_evals = np.concatenate((t1,t2,t3,t4,t5), axis=None)
t_evals = np.unique(t_evals)
capacity = param["Nominal cell capacity [A.h]"]
current = C_rates * capacity


def I_fun(A):
    def current(t):
        period = 3600 / 0.1
        rest_period = 3600
        k = 10
        rest = pybamm.sigmoid( t , rest_period, k)
        discharge = pybamm.sigmoid( t , rest_period + period, k) - rest
        charge = pybamm.sigmoid( rest_period + period, t, k)
        return A * discharge - A * charge
        # return A * (t < period) - A * (t >= period)
        # return A * 0.2 * pybamm.sin(2 * np.pi * t / 7200 )
    return current


param["Current function [A]"] = I_fun(current)

sim1 = pybamm.Simulation(
    model,
    parameter_values=param,
    solver=pybamm.CasadiSolver(dt_max = 10),
)
solution = sim1.solve(t_eval = t_evals)
stop = timeit.default_timer()
print("running time: " + str(stop - start) + "s")


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
