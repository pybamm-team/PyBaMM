#
# Simulate drive cycle loaded from csv file
#
import pybamm
import pandas as pd


pybamm.set_logging_level("INFO")

# load model and update parameters so the input current is the US06 drive cycle
model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})
param = model.default_parameter_values


# import drive cycle from file
data_loader = pybamm.DataLoader()
drive_cycle = pd.read_csv(
    data_loader.get_data("US06.csv"), comment="#", header=None
).to_numpy()

# create interpolant
current_interpolant = pybamm.Interpolant(drive_cycle[:, 0], drive_cycle[:, 1], pybamm.t)

# set drive cycle
param["Current function [A]"] = current_interpolant


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
        "Voltage [V]",
        "X-averaged cell temperature [K]",
    ]
)
