import pybamm

pybamm.set_logging_level("INFO")
experiment = pybamm.Experiment(
    [
        "Discharge at C/10 for 10 hours or until 3.3 V",
        "Rest for 1 hour",
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 50 mA",
        #    "Rest for 1 hour",
    ]
    * 1,
    period="30 minutes",
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment)

solvers = [
    pybamm.CasadiSolver(),
    # pybamm.CasadiSolver(return_event=True)
]
sols = []

for solver in solvers:
    sol = sim.solve(solver=solver)
    sols.append(sol)

c_n = sol["Negative particle surface concentration"]

pybamm.dynamic_plot(sols, ["Negative particle surface concentration"])
