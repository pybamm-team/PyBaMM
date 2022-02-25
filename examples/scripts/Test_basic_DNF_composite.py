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
param.update(
    {
        "Maximum concentration in negative electrode of phase 1 [mol.m-3]": 28700,
        "Initial concentration in negative electrode of phase 1 [mol.m-3]": 23600,
        "Negative electrode mass fraction of phase 1": 0.98,
        "Negative electrode diffusivity of phase 1 [m2.s-1]": 3.3e-14,
        "Negative electrode mass fraction of phase 2": 0.02,
        "Initial concentration in negative electrode of phase 2 [mol.m-3]": 270000,
        "Maximum concentration in negative electrode of phase 2 [mol.m-3]": 278000,
    }
)

# initial concentration of graphite 29866 [mol.m-3]
# -> 0.9134 * 25830 + 0.0866 * 275220

C_rates = 0.01
Q_cell = 4.89  # [Ah]
period = 3600 / C_rates * Q_cell / 5
rest_period = 3600
rest_period2 = 3600
transition = period / 400
timestep = [rest_period, period, rest_period2, period]
points = [101, 401, 101, 401]
t_evals = []
t0 = 0
for i in range(0, len(timestep)):
    t1 = np.linspace(t0, t0 + timestep[i] - transition, points[i])
    t0 += timestep[i]
    if i < len(timestep) - 1:
        t2 = np.linspace(t0 - transition, t0 + transition, 41)  # transition points
    else:
        t2 = []
    t_evals = np.concatenate((t_evals, t1, t2))

t_evals = np.unique(t_evals)
capacity = param["Nominal cell capacity [A.h]"]
I_load = C_rates * capacity


def I_fun(A):
    def current(t):
        C_rates = 0.01
        Q_cell = 4.89  # [Ah]
        period = 3600 / C_rates * Q_cell / 5
        rest_period = 3600
        rest_period2 = 3600
        k = 20
        rest = pybamm.sigmoid(t, rest_period, k)
        discharge = pybamm.sigmoid(t, rest_period + period, k) - rest
        # rest3 = pybamm.sigmoid( rest_period + period * 2 + rest_period2, t, k)
        charge = pybamm.sigmoid(rest_period + period + rest_period2, t, k)
        return A * discharge - A * charge
        # return A * (t < period) - A * (t >= period)
        # return A * 0.2 * pybamm.sin(2 * np.pi * t / 7200 )

    return current


param["Current function [A]"] = I_fun(I_load)

sim1 = pybamm.Simulation(
    model,
    parameter_values=param,
    solver=pybamm.CasadiSolver(dt_max=10),
)
solution = sim1.solve(t_eval=t_evals)
stop = timeit.default_timer()
print("running time: " + str(stop - start) + "s")

# %%
