import pybamm
import numpy as np

model = pybamm.lithium_sulfur.MarinescuEtAl2016()

param = model.default_parameter_values
param["Current function [A]"] = 1.7

solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6)

sim = pybamm.Simulation(model, parameter_values=param, solver=solver)

sim.step(3600, npts=3600)
sim.step(3480, npts=3480)

sim.plot(model.variables, time_unit="seconds")
