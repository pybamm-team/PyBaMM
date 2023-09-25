#
# Compare half-cell lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    pybamm.lithium_ion.SPM({"working electrode": "positive"}),
    pybamm.lithium_ion.SPMe({"working electrode": "positive"}),
    pybamm.lithium_ion.DFN({"working electrode": "positive"}),
]

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)
