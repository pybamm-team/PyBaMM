#
# Example showing how to initialize a model with another model.
#

import pybamm
import numpy as np
import pandas as pd

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.DFN()
# import drive cycle from file
drive_cycle = pd.read_csv(
    "pybamm/input/drive_cycles/US06.csv", comment="#", header=None
).to_numpy()
# create interpolant
param = model.default_parameter_values
timescale = param.evaluate(model.timescale)
current_interpolant = pybamm.Interpolant(drive_cycle, timescale * pybamm.t)
# set drive cycle
param["Current function [A]"] = current_interpolant
# create and run simulation using the CasadiSolver in "fast" mode, remembering to
# pass in the updated parameters
sim_US06_1 = pybamm.Simulation(
    model, parameter_values=param, solver=pybamm.CasadiSolver(mode="fast")
)
sol_US06_1 = sim_US06_1.solve()

# Charge with CCCV
experiment = pybamm.Experiment(
    ["Charge at 1 A until 4.1 V", "Hold at 4.1 V until 50 mA"]
)
sim_cccv = pybamm.Simulation(model, experiment=experiment)
sol_cccv = sim_cccv.solve()

# MODEL RE-INITIALIZATION: #############################################################
# Now initialize the model with the solution of the charge, and then discharge with
# the US06 drive cycle
# We could also do this inplace by setting inplace to True, which modifies the original
# model in place
new_model = model.set_initial_conditions_from(sol_cccv, inplace=False)
########################################################################################

sim_US06_2 = pybamm.Simulation(
    new_model, parameter_values=param, solver=pybamm.CasadiSolver(mode="fast")
)
sol_US06_2 = sim_US06_2.solve()

# Plot both solutions, we can clearly see the difference now that initial conditions
# have been updated
pybamm.dynamic_plot(
    [sol_US06_1, sol_US06_2], labels=["Default initial conditions", "Fully charged"]
)
