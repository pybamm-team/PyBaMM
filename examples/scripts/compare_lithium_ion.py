#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
# load models
models = [
    # pybamm.lithium_ion.SPM(),
    # pybamm.lithium_ion.SPMe(),
    pybamm.lithium_ion.DFN(),
    # pybamm.lithium_ion.DFN({"particle": "uniform profile"}),
    # pybamm.lithium_ion.DFN({"particle": "quadratic profile"}, name="2"),
    # pybamm.lithium_ion.DFN({"particle": "quartic profile"}, name="4"),
    # pybamm.lithium_ion.NewmanTobias(),
]

# create and run simulations
sims = []
for model in models:
    parameter_values = model.default_parameter_values
    parameter_values["Current function [A]"] = 0.0006
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)
