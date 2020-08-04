#
# Compare lithium-ion battery models with and without particle size distibution
#
import numpy as np
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    pybamm.lithium_ion.DFN(name="standard DFN"),
    pybamm.lithium_ion.DFN(name="particle DFN"),
]

# load parameter values
params = [models[0].default_parameter_values, models[1].default_parameter_values]


def negative_distribution(x):
    return 1 + 2 * x / models[1].param.l_n


def positive_distribution(x):
    return 1 + 2 * (1 - x) / models[1].param.l_p


params[1]["Negative particle distribution in x"] = negative_distribution
params[1]["Positive particle distribution in x"] = positive_distribution


# set up and solve simulations
t_eval = np.linspace(0, 3600, 100)
sols = []
for model, param in zip(models, params):
    sim = pybamm.Simulation(model, parameter_values=param)
    sol = sim.solve(t_eval)
    sols.append(sol)

output_variables = [
    "Negative particle surface concentration",
    "Electrolyte concentration",
    "Positive particle surface concentration",
    "Current [A]",
    "Negative electrode potential [V]",
    "Electrolyte potential [V]",
    "Positive electrode potential [V]",
    "Terminal voltage [V]",
    "Negative particle distribution in x",
    "Positive particle distribution in x",
]

# plot
plot = pybamm.QuickPlot(sols, output_variables=output_variables)
plot.dynamic_plot()
