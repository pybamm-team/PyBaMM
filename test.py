import pybamm

options = {
    "particle size": "distribution",
    "surface form": "algebraic",
}
model = pybamm.lithium_ion.SPMe(options)

param = pybamm.ParameterValues("Marquis2019")
param = pybamm.get_size_distribution_parameters(param)

sim = pybamm.Simulation(model, parameter_values=param)
sim.solve([0, 3600])
sim.plot()
