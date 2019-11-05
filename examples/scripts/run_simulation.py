import pybamm

model = pybamm.lithium_ion.SPM()

sim = pybamm.Simulation(model)
sim.solve()
sim.plot()
