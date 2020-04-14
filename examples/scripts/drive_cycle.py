#
# Simulate drive cycle loaded from csv file
#
import pybamm

pybamm.set_logging_level("INFO")

# load model and update parameters so the input current is the US06 drive cycle
model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})
param = model.default_parameter_values
param["Current function [A]"] = "[current data]US06"

# create and run simulation using the CasadiSolver in "fast" mode, remembering to
# pass in the updated parameters
sim = pybamm.Simulation(
    model, parameter_values=param, solver=pybamm.CasadiSolver(mode="fast")
)
sim.solve()
sim.plot(
    [
        "Negative particle surface concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "Positive particle surface concentration [mol.m-3]",
        "Current [A]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        "Terminal voltage [V]",
        "X-averaged cell temperature",
    ]
)
