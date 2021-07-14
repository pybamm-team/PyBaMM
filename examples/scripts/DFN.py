import pybamm

pybamm.set_logging_level("INFO")
options = {"cell geometry": "pouch", "thermal": "x-lumped", "dimensionality": 2}
model = pybamm.lithium_ion.SPMe(options=options)
experiment = pybamm.Experiment(["Discharge with 5 C for 1 minute"])
sim = pybamm.Simulation(model, experiment=experiment)
sim.solve(solver=pybamm.CasadiSolver("fast with events"))
