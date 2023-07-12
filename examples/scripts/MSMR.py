import pybamm

model = pybamm.lithium_ion.SPM(
    {
        "open-circuit potential": "MSMR",
        "particle": "MSMR",
    }
)


parameter_values = pybamm.ParameterValues("MSMR_Example")
experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C until 3V",
            "Rest for 1 hour",
            "Charge at C/2 until 4.1 V",
            "Hold at 4.1 V until 10 mA",
            "Rest for 1 hour",
        ),
    ]
)
sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
sim.solve(initial_soc=0.9)
sim.plot()
