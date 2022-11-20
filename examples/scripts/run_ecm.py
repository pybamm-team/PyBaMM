import pybamm
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

model = pybamm.ecm.EquivalentCircuitModel()

experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/10 for 10 hours or until 3.3 V",
            "Rest for 1 hour",
            "Charge at 100 A until 4.1 V (1 second period)",
            "Hold at 4.1 V until 5 A (1 seconds period)",
            "Rest for 1 hour",
        ),
    ]
    * 3
)

sim = pybamm.Simulation(model, experiment=experiment)
sim.solve()
sim.plot()
