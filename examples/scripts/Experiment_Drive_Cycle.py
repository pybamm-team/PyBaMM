#
# Constant-current constant-voltage charge with US06 Drive Cycle using Experiment Class.
#
import pybamm
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir(pybamm.__path__[0] + "/..")

pybamm.set_logging_level("INFO")

# import drive cycle from file
drive_cycle = pd.read_csv(
    "pybamm/input/drive_cycles/US06.csv", comment="#", header=None
).to_numpy()

experiment = pybamm.Experiment(
    [
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 50 mA",
        "Rest for 1 hour",
        "Run US06",
        "Rest for 1 hour",
    ]* 3
    , drive_cycles={"US06": drive_cycle}
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()

# Show all plots
sim.plot()