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
    * 3
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()

# Plot voltages from the discharge segments only
fig, ax = plt.subplots()
for i in range(3):
    # Extract sub solutions
    sol = sim.solution.sub_solutions[i * 5]
    # Extract variables
    t = sol["Time [h]"].entries
    V = sol["Terminal voltage [V]"].entries
    # Plot
    ax.plot(t - t[0], V, label="Discharge {}".format(i + 1))
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Voltage [V]")
    ax.set_xlim([0, 13])
ax.legend()

# Save time, voltage, current, discharge capacity and temperature to csv and matlab
# formats
sim.solution.save_data(
    "output.mat",
    [
        "Time [h]",
        "Current [A]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [K]",
    ],
    to_format="matlab",
)
sim.solution.save_data(
    "output.csv",
    [
        "Time [h]",
        "Current [A]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [K]",
    ],
    to_format="csv",
)

# Show all plots
sim.plot()
