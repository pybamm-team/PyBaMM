import pybamm as pb

pb.set_logging_level("DEBUG")
pb.settings.debug_mode = True

options = {"sei": "reaction limited"}
model = pb.lithium_ion.DFN(options)

sim = pb.Simulation(model)

solver = pb.CasadiSolver(mode="safe")
sim.solve(solver=solver)
sim.plot(["Terminal voltage [V]", "Total SEI thickness [m]"])
