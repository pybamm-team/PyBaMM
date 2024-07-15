import pybamm

pybamm.set_logging_level("INFO")

model = pybamm.equivalent_circuit.Thevenin(options={"diffusion element": "true"})
parameter_values = model.default_parameter_values

parameter_values.update(
    {"Diffusion time constant [s]": 580}, check_already_exists=False
)

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

sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)
sim.solve()
sim.plot()
