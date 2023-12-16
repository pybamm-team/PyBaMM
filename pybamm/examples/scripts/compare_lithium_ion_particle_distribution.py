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


def negative_radius(x):
    "Negative particle radius as a function of through-cell position (x_n [m])"
    R_n_0 = params[0]["Negative particle radius [m]"]
    grading = 1 + 2 * x / models[1].param.n.L
    return grading * R_n_0


def positive_radius(x):
    "Positive particle radius as a function of through-cell position (x_p [m])"
    R_p_0 = params[0]["Positive particle radius [m]"]
    grading = 1 + 2 * (models[1].param.L_x - x) / models[1].param.p.L
    return grading * R_p_0


params[1]["Negative particle radius [m]"] = negative_radius
params[1]["Positive particle radius [m]"] = positive_radius

# set up and solve simulations
t_eval = np.linspace(0, 3600, 100)
sols = []
for model, param in zip(models, params):
    sim = pybamm.Simulation(model, parameter_values=param)
    sol = sim.solve(t_eval)
    sols.append(sol)

output_variables = [
    "Negative particle surface concentration",
    "Electrolyte concentration [Molar]",
    "Positive particle surface concentration",
    "Current [A]",
    "Negative electrode potential [V]",
    "Electrolyte potential [V]",
    "Positive electrode potential [V]",
    "Voltage [V]",
    "Negative particle radius",
    "Positive particle radius",
]

# plot
plot = pybamm.QuickPlot(sols, output_variables=output_variables)
plot.dynamic_plot()
