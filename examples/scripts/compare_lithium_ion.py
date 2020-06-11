#
# Compare lithium-ion battery models
#
import pybamm

# load models
models = [
    pybamm.lithium_ion.SPM(),
    pybamm.lithium_ion.SPMe(),
    pybamm.lithium_ion.DFN(),
]

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model)
    sim.solve()
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)