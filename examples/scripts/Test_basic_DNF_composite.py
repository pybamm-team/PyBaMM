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
param.update({"Upper voltage cut-off [V]": 4.21})
param.update({"Lower voltage cut-off [V]": 2.49})

C_rates = 0.5
t_evals = np.linspace(0, 7200*2, 200)
capacity = param["Nominal cell capacity [A.h]"]
current = C_rates * capacity


def I_fun(A):
    def current(t):
        # if t < 36:
        #     I_result = A 
        # else:
        #     I_result = - A 
        # return I_result 
        return A * 0.2 * pybamm.sin(2 * np.pi * t / 7200 )
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
    ]
)
plot.dynamic_plot()
stop = timeit.default_timer()
print("running time: " + str(stop - start) + "s")


# %%
