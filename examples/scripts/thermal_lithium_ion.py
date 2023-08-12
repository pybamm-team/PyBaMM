#
# Compares the full and lumped thermal models for a single layer Li-ion cell
#

import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    pybamm.lithium_ion.SPMe({"thermal": "x-full"}),
    pybamm.lithium_ion.SPMe({"thermal": "x-lumped"}),
]

# load parameter values and update cooling coefficients
parameter_values = pybamm.ParameterValues("Marquis2019")
parameter_values.update(
    {
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 5,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 5,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 10,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 10,
        "Edge heat transfer coefficient [W.m-2.K-1]": 5,
    }
)

# create and solve simulations
sols = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sol = sim.solve([0, 3600])
    sols.append(sol)

# plot
output_variables = [
    "Voltage [V]",
    "X-averaged cell temperature [K]",
    "Cell temperature [K]",
]
labels = ["Full thermal model", "Lumped thermal model"]
pybamm.dynamic_plot(sols, output_variables, labels=labels)
