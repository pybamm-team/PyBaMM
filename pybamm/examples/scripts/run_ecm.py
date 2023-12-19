import pybamm

pybamm.set_logging_level("INFO")

model = pybamm.equivalent_circuit.Thevenin()

experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/10 for 10 hours or until 3.3 V",
            "Rest for 30 minutes",
            "Rest for 2 hours",
            "Charge at 100 A until 4.1 V",
            "Hold at 4.1 V until 5 A",
            "Rest for 30 minutes",
            "Rest for 1 hour",
        ),
    ]
)

sim = pybamm.Simulation(model, experiment=experiment)
sim.solve()
sim.plot()
