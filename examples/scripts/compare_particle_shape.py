#
# Example showing how to prescribe the surface area to volume ratio independent of
# the assumed particle shape. Setting the "particle shape" option to "user" returns
# a model which solves a spherical diffusion problem in the particles, but passes
# a user supplied surface area to volume ratio
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

models = [
    pybamm.lithium_ion.DFN({"particle shape": "spherical"}, name="spherical"),
    pybamm.lithium_ion.DFN({"particle shape": "user"}, name="user"),
]
params = [models[0].default_parameter_values, models[0].default_parameter_values]

# set up and solve simulations
solutions = []
t_eval = np.linspace(0, 3600, 100)

for model, param in zip(models, params):
    if model.name == "user":
        # add the user supplied parameters
        param.update(
            {
                "Negative electrode surface area to volume ratio [m-1]": 170000,
                "Positive electrode surface area to volume ratio [m-1]": 200000,
            },
            check_already_exists=False,
        )

    sim = pybamm.Simulation(model, parameter_values=param)
    solution = sim.solve(t_eval)
    solutions.append(solution)

# plot solutions
pybamm.dynamic_plot(
    solutions,
    [
        "Negative particle surface concentration [mol.m-3]",
        "Positive particle surface concentration [mol.m-3]",
        "Negative electrode interfacial current density [A.m-2]",
        "Positive electrode interfacial current density [A.m-2]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        "Terminal voltage [V]",
    ],
)
