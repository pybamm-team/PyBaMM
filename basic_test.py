import pybamm

models = [
    # pybamm.lithium_ion.DFN(
    #    options={"particle": "fast diffusion"}, name="fastDFN particle"
    # ),
    pybamm.lithium_ion.DFN(name="DFN particle"),
    pybamm.lithium_ion.DFN(name="DFN"),
]
parameters = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)


def negative_distribution(x):
    return 1 + x


def positive_distribution(x):
    return 1 + (x - (1 - models[1].param.l_p))


parameters["Negative particle distribution in x"] = negative_distribution
parameters["Positive particle distribution in x"] = positive_distribution
parameters["Negative electrode diffusivity [m2.s-1]"] = 1e-1
parameters["Positive electrode diffusivity [m2.s-1]"] = 1e-1
sols = []

for model in models:
    if model.name == "DFN":
        params = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)
        params["Negative electrode diffusivity [m2.s-1]"] = 1e-1
        params["Positive electrode diffusivity [m2.s-1]"] = 1e-1
    else:
        params = parameters
    sim = pybamm.Simulation(model, parameter_values=params)
    sol = sim.solve([0, 3600])
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

pybamm.dynamic_plot(sols, output_variables=output_variables)
