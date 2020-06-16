#
# Compare lead-acid battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    pybamm.lead_acid.LOQS(),
    pybamm.lead_acid.FOQS(),
    pybamm.lead_acid.Composite(),
    pybamm.lead_acid.Full(),
]

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model)
    sim.solve()
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)
