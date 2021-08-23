import pybamm as pb
import numpy as np

pb.set_logging_level("INFO")

model = pb.lithium_ion.SPM({"SEI": "reaction limited", "SEI porosity change": "true"})

parameter_values = model.default_parameter_values

parameter_values["Current function [A]"] = 0

sim = pb.Simulation(model, parameter_values=parameter_values)

solver = pb.CasadiSolver(mode="fast")

years = 30
days = years * 365
hours = days * 24
minutes = hours * 60
seconds = minutes * 60

t_eval = np.linspace(0, seconds, 100)

sim.solve(t_eval=t_eval, solver=solver)

pb.dynamic_plot(sim, ["X-averaged negative electrode porosity"])
