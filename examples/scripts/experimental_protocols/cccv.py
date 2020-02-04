#
# Constant-current constant-voltage charge
#
import pybamm

pybamm.set_logging_level("INFO")
experiment = pybamm.Experiment(
    [
        "Discharge at C/10 until 3.3 V",
        "Rest for 1 hour",
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 50 mA",
        "Rest for 1 hour",
    ]
    * 3,
    {},
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()
sim.plot()
