import pybamm

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

parameter_values = model.default_parameter_values

parameter_values.update({
    "Initial SoC": 0.5,
    "Initial cell temperature [degC]": 25,
    "Initial jig temperature [degC]": 25,
    "Cell capacity [A.h]": 100,
    "Nominal cell capacity [A.h]": 100,
    "Ambient temperature [degC]": 25,
    "Current function [A]": 100,
    "Upper voltage cut-off [V]": 4.2,
    "Lower voltage cut-off [V]": 3.2,
    "Cell thermal mass [J/K]": 1000,
    "Cell-jig heat transfer coefficient [W/K]": 10,
    "Jig thermal mass [J/K]": 500,
    "Jig-air heat transfer coefficient [W/K]": 10,
    "R0 [Ohm]": 0.4e-3,
    "Element-1 initial overpotential [V]": 0,
    "R1 [Ohm]": 0.6e-3,
    "C1 [F]": 30 / 0.6e-3,
    "Entropic change [V/K]": 0,
    "RCR lookup limit [A]": 340,
})

solver = pybamm.CasadiSolver(mode="safe")
sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values, solver=solver)
sim.solve(calc_esoh=False)
sim.plot()
