import pybamm
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

models = [
    pybamm.lithium_sulfur.OriginalMarinescuEtAl2016(name="original"),
    pybamm.lithium_sulfur.xOriginalMarinescuEtAl2016(name="xoriginal"),
    pybamm.lithium_sulfur.MarinescuEtAl2016(name="original with base li-s"),
]

period = 1
discharge = ["Discharge at 1.7A for 1.8 hours"]
experiment = pybamm.Experiment(discharge, period=f"{period} seconds")

sims = []
for model in models:
    sim = pybamm.Simulation(model, experiment=experiment)
    sim.solve()
    sims.append(sim)
pybamm.dynamic_plot(sims)
