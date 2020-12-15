import pybamm

pybamm.set_logging_level("INFO")
experiment = pybamm.Experiment(["Discharge at 1C until 3.6V", "Rest for 1 hour"] * 2)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment)

solvers = [pybamm.CasadiSolver(), pybamm.CasadiSolver(return_event=True)]
sols = []
times = []
for solver in solvers:
    sol = sim.solve(solver=solver)
    sols.append(sol)
    times.append(sol["Time [s]"].entries)

pybamm.dynamic_plot(
    sols, ["Terminal voltage [V]"], labels=["return_event=False", "return_event=True"]
)
