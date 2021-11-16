#
# Example showing how to change the interface utilisation
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

models = [
    pybamm.lithium_ion.DFN({"interface utilisation": "full"}, name="full utilisation"),
    pybamm.lithium_ion.DFN(
        {"interface utilisation": "constant"}, name="constant utilisation"
    ),
]
params = [models[0].default_parameter_values, models[0].default_parameter_values]

# set up and solve simulations
solutions = []
t_eval = np.linspace(0, 3600, 100)

for model, param in zip(models, params):
    if model.name == "constant utilisation":
        # add the user supplied parameters
        param.update(
            {
                "Initial negative electrode interface utilisation": 0.9,
                "Initial positive electrode interface utilisation": 0.8,
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
