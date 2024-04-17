import pybamm
import numpy as np

# create the model
model = pybamm.lithium_ion.DFN()

# set the default model parameters
param = model.default_parameter_values

# change the current function to be an input parameter
param["Current function [A]"] = "[input]"

simulation = pybamm.Simulation(model, parameter_values=param)

# solve the model at the given time points, passing multiple current values as inputs
t_eval = np.linspace(0, 600, 300)
inputs = [{"Current function [A]": x} for x in range(1, 3)]
sol = simulation.solve(t_eval, inputs=inputs)
