#
# Compare different models for intercalation kinetics.
#
from __future__ import annotations

import pybamm

pybamm.set_logging_level("INFO")

# load models
kinetics = [
    "symmetric Butler-Volmer",
    "asymmetric Butler-Volmer",
    "linear",
    "Marcus-Hush-Chidsey",
]

extra_parameters = [
    {},
    {
        "Negative electrode Butler-Volmer transfer coefficient": 0.6,
        "Positive electrode Butler-Volmer transfer coefficient": 0.6,
    },
    {},
    {
        "Negative electrode reorganization energy [eV]": 0.35,
        "Positive electrode reorganization energy [eV]": 0.35,
        "Positive electrode exchange-current density [A.m-2]": 5,
    },
]
# create and run simulations
sims = []
for k, p in zip(kinetics, extra_parameters):
    model = pybamm.lithium_ion.DFN({"intercalation kinetics": k}, name=k)
    param = model.default_parameter_values
    param.update(p, check_already_exists=False)
    sim = pybamm.Simulation(model, parameter_values=param)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)
