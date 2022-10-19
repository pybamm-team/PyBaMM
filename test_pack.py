import pybamm
import numpy as np

model = pybamm.lithium_ion.DFN(name="DFN")
parameter_values = model.default_parameter_values
parameter_values.update({"Current function [A]":pybamm.PsuedoInputParameter("cell_current")})
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sim.build()


