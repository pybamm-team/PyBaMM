import pybamm

model = pybamm.ecm.EquivalentCircuitModel()

sim = pybamm.Simulation(model)
sim.solve([0, 100])

# sol = sim.solution

# print("hi")
