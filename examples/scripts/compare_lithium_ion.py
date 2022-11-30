#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("DEBUG")
# load models
models = [
    # pybamm.lithium_ion.SPM(),
    # pybamm.lithium_ion.SPMe(),
    pybamm.lithium_ion.BasicDFN(),
    # pybamm.lithium_ion.DFN(
    #     {"particle": "uniform profile"}
    # ),
    # pybamm.lithium_ion.DFN({"particle": "quadratic profile"}, name="2"),
    # pybamm.lithium_ion.DFN({"particle": "quartic profile"}, name="4"),
    # pybamm.lithium_ion.NewmanTobias(),
]

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver("fast"))
    sim.solve([0, 4000])
    sims.append(sim)

# plot
# pybamm.dynamic_plot(sims)
