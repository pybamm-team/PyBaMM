#
# Constant-current constant-voltage charge
#
import pybamm
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")
experiment = pybamm.Experiment(
    [
        "Discharge at C/10 for 10 hours or until 3.3 V",
        "Rest for 1 hour",
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 50 mA",
        "Rest for 1 hour",
    ]
    * 3,
    period="2 minutes",
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()

# Plot voltages from the discharge segments only
for i in [0, 5, 10]:
    sol = sim.solution.sub_solutions[i]
    t = sol["Time [h]"].entries
    V = sol["Terminal voltage [V]"].entries
    plt.plot(t - t[0], V, label="Run {}".format(i))
plt.legend()

# Show all plots
sim.plot()
