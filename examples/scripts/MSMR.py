import pybamm
from MSMR_example import get_parameter_values

pybamm.set_logging_level("DEBUG")
model = pybamm.lithium_ion.DFN({"open-circuit potential": "MSMR", "particle": "MSMR"})
parameter_values = pybamm.ParameterValues(get_parameter_values())
experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C for 1 hour or until 3 V",
            # "Rest for 1 hour",
            # "Charge at C/3 until 4 V",
            # "Hold at 4 V until 10 mA",
            # "Rest for 1 hour",
        ),
    ]
)
sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
sim.solve(calc_esoh=True)
sim.plot(
    # [
    #    "Negative electrode open-circuit potential [V]",
    #    "Positive electrode open-circuit potential [V]",
    # ]
)
