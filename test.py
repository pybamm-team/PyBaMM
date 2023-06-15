import pybamm
import numpy as np
import matplotlib.pyplot as plt

models = [
    pybamm.lithium_ion.DFN(
        {"cell geometry": "pouch", "thermal": "x-full"}, name="x-full"
    ),
    pybamm.lithium_ion.DFN(
        {"cell geometry": "pouch", "thermal": "lumped"}, name="lumped"
    ),
]
parameter_values = pybamm.ParameterValues("Marquis2019")
parameter_values.update(
    {
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Total heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Negative current collector density [kg.m-3]": 8954.0,
        "Positive current collector density [kg.m-3]": 2707.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
    }
)
parameter_values.set_initial_stoichiometries(1)
parameter_values["Current function [A]"] = parameter_values[
    "Nominal cell capacity [A.h]"
]
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve(np.linspace(0, 1800, 100))
    sims.append(sim)
plot = pybamm.dynamic_plot(
    sims,
    [
        "Cell temperature [K]",
        "X-averaged cell temperature [K]",
        "X-averaged total heating [W.m-3]",
    ],
)
