import pybamm

pybamm.set_logging_level("INFO")

# TODO: check reversible heat generation

options = {"number of rc elements": 2}
model = pybamm.equivalent_circuit.Thevenin(options=options)

parameter_values = model.default_parameter_values
parameter_values.update(
    {
        "R2 [Ohm]": 0.3e-3,
        "C2 [F]": 1000 / 0.3e-3,
        "Element-2 initial overpotential [V]": 0,
    },
    check_already_exists=False,
)

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
)

sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)
sim.solve()
sim.plot()
