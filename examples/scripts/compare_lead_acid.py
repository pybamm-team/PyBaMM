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
    pybamm.lead_acid.Full({"surface form": "differential"}),
]

# create and run simulations
sims = []
for model in models:
    model.convert_to_format = None
    sim = pybamm.Simulation(model)
    sim.solve([0, 10])
    sims.append(sim)

# plot
pybamm.dynamic_plot(
    sims, ["Sum of negative electrode volumetric interfacial current densities"]
)
