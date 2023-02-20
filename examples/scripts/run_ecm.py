import pybamm

pybamm.set_logging_level("INFO")

model = pybamm.equivalent_circuit.Thevenin()

experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/10 for 10 hours or until 3.3 V at 15oC",
            "Rest for 30 minutes at 15oC",
            "Rest for 2 hours at 35oC",
            "Charge at 100 A until 4.1 V at 35oC (1 second period)",
            "Hold at 4.1 V until 5 A at 35oC (1 seconds period)",
            "Rest for 30 minutes at 35oC",
            "Rest for 1 hour at 25oC",
        ),
    ]
)

sim = pybamm.Simulation(model, experiment=experiment)
sim.solve()
sim.plot()
