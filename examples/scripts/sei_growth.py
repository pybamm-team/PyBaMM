import pybamm as pb

pb.set_logging_level("DEBUG")
# pb.settings.debug_mode = True

options = {"sei": "reaction limited"}
model = pb.lithium_ion.DFN(options)

experiment = pb.Experiment(["Rest for 100 hours"])

parameter_values = model.default_parameter_values

sim = pb.Simulation(model, experiment=experiment)

solver = pb.CasadiSolver(mode="safe")

sim.solve(solver=solver)
sim.plot(
    [
        "Terminal voltage [V]",
        "X-averaged total SEI thickness",
        "Loss of lithium to SEI [mols]",
    ]
)
