#
# Simulate drive cycle loaded from csv file
#
import pybamm

# load model and update param with the input current provided by the US06 drive cycle
model = pybamm.lithium_ion.DFN()
param = model.default_parameter_values
param["Current function [A]"] = "[current data]US06"

# create and run simulation using the CasadiSolver in "fast" mode, remebering to
# pass in the updated parameters
sim = pybamm.Simulation(
    model, parameter_values=param, solver=pybamm.CasadiSolver(mode="fast")
)
sim.solve()
sim.plot()
