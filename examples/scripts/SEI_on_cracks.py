import pybamm
import matplotlib.pyplot as plt
import numpy as np

pybamm.set_logging_level("DEBUG")
parameter_values = pybamm.ParameterValues("OKane2022")
model1 = pybamm.lithium_ion.DFN({"SEI": "solvent-diffusion limited"})
experiment = pybamm.Experiment(["Discharge at 1C until 2.5 V"])
sim1 = pybamm.Simulation(model1, parameter_values=parameter_values, experiment=experiment)
sol1 = sim1.solve()