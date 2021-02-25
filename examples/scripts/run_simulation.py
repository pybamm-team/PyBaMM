import pybamm

model = pybamm.lithium_ion.SPM()

sim = pybamm.Simulation(model)
sim.solve([0, 3600])
sim.plot()
