import pybamm

pybamm.set_logging_level("INFO")

models = [
    pybamm.lithium_sulfur.OriginalMarinescuEtAl2016(),
    pybamm.lithium_sulfur.MarinescuEtAl2016(),
]

sims = []
for model in models:
    solver = pybamm.CasadiSolver(mode="fast")
    sim = pybamm.Simulation(model, solver=solver)
    sim.solve([0, 3.06 * 3600])
    sims.append(sim)
pybamm.dynamic_plot(sims)

experiment = pybamm.Experiment(
    ["Discharge at 1A for 10 minutes", "Rest for 5 minutes"] * 3
)
models = [
    pybamm.lithium_sulfur.OriginalMarinescuEtAl2016(),
    pybamm.lithium_sulfur.MarinescuEtAl2016(),
]
sims = []
for model in models:
    sim = pybamm.Simulation(model, experiment=experiment)
    sim.solve()
    sims.append(sim)
pybamm.dynamic_plot(sims)
