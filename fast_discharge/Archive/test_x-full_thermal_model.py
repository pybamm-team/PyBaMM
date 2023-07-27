#
# Simulate drive cycle loaded from csv file
#
import pybamm
import pandas as pd
import os

os.chdir(pybamm.__path__[0] + "/..")

pybamm.set_logging_level("INFO")

# load model and update parameters so the input current is the US06 drive cycle
model = pybamm.lithium_ion.SPMe(options = {"thermal": "x-full"})
param = model.default_parameter_values


param.update({
   "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 1000,
   "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 0,
    "Negative electrode specific heat capacity [J.kg-1.K-1]": 1100,
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 110,
    "Edge heat transfer coefficient [W.m-2.K-1]":0,
})


# import drive cycle from file
# drive_cycle = pd.read_csv(
#     "pybamm/input/drive_cycles/US06.csv", comment="#", header=None
# ).to_numpy()
# drive_cycle[:,1] = drive_cycle[:,1]*2

experiment = pybamm.Experiment(
        [
        # "Run US06 (A) until 3.0 V"
        "Discharge at 5C for 5 minutes or until 3.0 V "
        ],
        # drive_cycles={
        #     'US06': drive_cycle
        # }
    )

# create interpolant
# timescale = param.evaluate(model.timescale)


# create and run simulation using the CasadiSolver in "fast" mode, remembering to
# pass in the updated parameters
var_pts=model.default_var_pts
var_pts.update({"x_n": 10, "x_s": 5, "x_p": 10})
sim = pybamm.Simulation(
    model, parameter_values=param, experiment= experiment, solver=pybamm.CasadiSolver(mode="fast"), var_pts=var_pts
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
        "Cell temperature [K]",
    ]
)
