#
# Constant-current constant-voltage charge
#
import pybamm

pybamm.set_logging_level("NOTICE")
experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C until 2.5 V",
            "Rest for 1 hour",
            "Charge at 5 A until 4.2 V",
            "Hold at 4.2 V until 10 mA",
            "Rest for 1 hour",
        ),
    ]
    * 3
)
model = pybamm.lithium_ion.BasicDFN()
parameter_values = pybamm.ParameterValues("Chen2020")

sim = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=parameter_values,
)
sim.solve()
sim.plot()
