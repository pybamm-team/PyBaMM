#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
options = {
    "particle phases": ("2", "1"),
    "open circuit potential": (("single", "current sigmoid"), "single"),
}
models = [
    # pybamm.lithium_ion.SPM(options),
    # pybamm.lithium_ion.SPMe(options),
    # pybamm.lithium_ion.BasicDFNComposite(),
    pybamm.lithium_ion.DFN({"particle size": "distribution"}),
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
pybamm.dynamic_plot(sims, ["Terminal voltage [V]"])
