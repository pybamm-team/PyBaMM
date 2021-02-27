#
# GITT discharge
#
import pybamm

pybamm.set_logging_level("VERBOSE")
experiment = pybamm.Experiment(
    [("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 20,
)
model = pybamm.lithium_ion.SPM()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()
sim.plot()
