import pybamm

model = pybamm.ecm.EquivalentCircuitModel()

sim = pybamm.Simulation(model)
sim.solve([0, 3600])

sim.plot()
# sol = sim.solution

# print("hi")
