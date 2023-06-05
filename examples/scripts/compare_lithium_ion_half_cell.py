#
# Compare half-cell lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    pybamm.lithium_ion.SPM({"half-cell": "true"}),
    pybamm.lithium_ion.SPMe({"half-cell": "true"}),
    pybamm.lithium_ion.DFN({"half-cell": "true"}),
]

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)
