#
# Simulate drive cycle loaded from csv file
#
import pybamm

pybamm.set_logging_level("INFO")

# load model and update parameters so the input current is the US06 drive cycle
model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Mohtat2020)
param["Current function [A]"] = "[current data]Cell32_cycle"
param["Electrolyte conductivity [S.m-1]"] = 2.108e-1
param["Electrolyte diffusivity [m2.s-1]"] = 9.118e-9
param["Negative electrode reaction rate"] = 3.471e-6


# create and run simulation using the CasadiSolver in "fast" mode, remembering to
# pass in the updated parameters
sim = pybamm.Simulation(
    model, parameter_values=param, solver=pybamm.CasadiSolver(mode="fast")
)
sim.solve()
sim.plot(
    [
        # "Negative particle surface concentration [mol.m-3]",
        # "Electrolyte concentration [mol.m-3]",
        # "Positive particle surface concentration [mol.m-3]",
        "Current [A]",
        # "Negative electrode potential [V]",
        # "Electrolyte potential [V]",
        # "Positive electrode potential [V]",
        "Terminal voltage [V]",
        # "X-averaged cell temperature",
    ]
)

# -- export to matlab file --
from scipy.io import savemat

savemat(
    "pybamm_outputs.mat",
    {
        "Current": sim.solution["Current [A]"].entries,
        "Time": sim.solution["Time [s]"].entries,
        "Voltage": sim.solution["Terminal voltage [V]"].entries,
        "Capacity": -sim.solution["Discharge capacity [A.h]"].entries,
        "c_ss_n": sim.solution["Negative particle surface concentration"].entries,
    },
)