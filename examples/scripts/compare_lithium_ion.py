#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    pybamm.lithium_ion.SPM({"particle phases": ("2", "1")}),
    pybamm.lithium_ion.SPMe({"particle phases": ("2", "1")}),
    pybamm.lithium_ion.DFN({"particle phases": ("2", "1")}),
    # pybamm.lithium_ion.NewmanTobias(),
]

param = pybamm.ParameterValues("Chen2020_composite")

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=param)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)
