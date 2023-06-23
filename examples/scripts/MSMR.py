import pybamm

model = pybamm.lithium_ion.DFN(
    {
        "open-circuit potential": "MSMR",
        "particle": "MSMR",
    }
)


parameter_values = pybamm.ParameterValues("MSMR_Example")
experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C for 1 hour or until 3 V",
            "Rest for 1 hour",
            "Charge at C/3 until 4 V",
            "Hold at 4 V until 10 mA",
            "Rest for 1 hour",
        ),
    ]
)
sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
sim.solve()
sim.plot()
