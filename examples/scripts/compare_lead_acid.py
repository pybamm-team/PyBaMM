#
# Compare lead-acid battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [pybamm.lead_acid.LOQS(), pybamm.lead_acid.Full()]

# create and run simulations
sims = []
for model in models:
    model.convert_to_format = None
    sim = pybamm.Simulation(model)
    sim.solve(
        [0, 3600 * 17],
    )
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)
