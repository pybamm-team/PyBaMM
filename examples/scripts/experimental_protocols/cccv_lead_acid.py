#
# Constant-current constant-voltage charge
#
import pybamm

pybamm.set_logging_level("INFO")
experiment = pybamm.Experiment(
    [
        "Discharge at C/2 until 11 V",
        "Rest for 1 hour",
        "Charge at C/2 until 14.5 V",
        "Hold at 14.5 V until 200 mA",
        "Rest for 1 hour",
    ]
)
model = pybamm.lead_acid.Full()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()
sim.plot()
