import pybamm
from MSMR_example import get_parameter_values

pybamm.set_logging_level("DEBUG")
model = pybamm.lithium_ion.SPM({"open-circuit potential": "MSMR", "particle": "MSMR"})
parameter_values = pybamm.ParameterValues(get_parameter_values())
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sim.solve([0, 3000])
sim.plot()
