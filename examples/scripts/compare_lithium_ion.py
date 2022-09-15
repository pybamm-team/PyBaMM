#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    # pybamm.lithium_ion.SPM({"thermal": "x-lumped"}),
    pybamm.lithium_ion.SPMe({"electrolyte conductivity": "integrated"}),
    # pybamm.lithium_ion.DFN(),
    # pybamm.lithium_ion.NewmanTobias(),
]

# create and run simulations
sims = []
for model in models:
    model.check_well_posedness()
    sim = pybamm.Simulation(model)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)
