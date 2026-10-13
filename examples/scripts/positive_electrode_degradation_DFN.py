import pybamm

pybamm.settings.tolerances["reg_power"] = 0

model = pybamm.lithium_ion.DFN({"positive electrode degradation": "true"})
param = pybamm.ParameterValues("Zhuo2023")

experiment = pybamm.Experiment(
    [
        (
            "Charge at 1.0 C until 4.2 V",
            "Hold at 4.2 V until C/100",
            "Rest for 30 minutes",
        )
    ]
)

sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
sim.solve(calc_esoh=False)
sim.plot(["Positive shell concentration of oxygen [mol.m-3]"], time_unit="minutes")
