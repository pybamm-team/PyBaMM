#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    # pybamm.lithium_ion.BasicSPM(),
    # pybamm.lithium_ion.SPM(),
    # pybamm.lithium_ion.SPMe({"electrolyte conductivity": "integrated"}),
    pybamm.lithium_ion.BasicDFN(),
    # pybamm.lithium_ion.NewmanTobias(),
]

# create and run simulations
sims = []
for model in models:
    model.events = []
    sim = pybamm.Simulation(model, C_rate=1)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)
