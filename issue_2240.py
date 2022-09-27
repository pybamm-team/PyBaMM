import pybamm
pybamm.logger.setLevel(5)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)
sim.build()


sim.solve([0, 3600])