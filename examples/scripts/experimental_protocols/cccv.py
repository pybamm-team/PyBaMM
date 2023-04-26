#
# Constant-current constant-voltage charge
#
import pybamm
import matplotlib.pyplot as plt

pybamm.set_logging_level("NOTICE")
experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C until 2.5 V",
            "Rest for 1 hour",
            "Charge at 5 A until 4.2 V",
            "Hold at 4.2 V until 10 mA",
            "Rest for 1 hour",
        ),
    ]
    * 3
)
model = pybamm.lithium_ion.DFN({"SEI": "ec reaction limited"})
parameter_values = pybamm.ParameterValues("Chen2020")

sim = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=parameter_values,
    solver=pybamm.CasadiSolver("fast with events"),
)
sim.solve()

# Plot voltages from the discharge segments only
fig, ax = plt.subplots()
for i in range(3):
    # Extract sub solutions
    sol = sim.solution.cycles[i]
    # Extract variables
    t = sol["Time [h]"].entries
    V = sol["Voltage [V]"].entries
    # Plot
    ax.plot(t - t[0], V, label="Discharge {}".format(i + 1))
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Voltage [V]")
    ax.set_xlim([0, t[-1] - t[0]])
ax.legend(loc="lower left")

# Save time, voltage, current, discharge capacity, temperature, and electrolyte
# concentration to csv and matlab formats
sim.solution.save_data(
    "output.mat",
    [
        "Time [h]",
        "Current [A]",
        "Voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [K]",
        "Electrolyte concentration [mol.m-3]",
    ],
    to_format="matlab",
    short_names={
        "Time [h]": "t",
        "Current [A]": "I",
        "Voltage [V]": "V",
        "Discharge capacity [A.h]": "Q",
        "X-averaged cell temperature [K]": "T",
        "Electrolyte concentration [mol.m-3]": "c_e",
    },
)
# We can only save 0D variables to csv
sim.solution.save_data(
    "output.csv",
    [
        "Time [h]",
        "Current [A]",
        "Voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [K]",
    ],
    to_format="csv",
)

# Show all plots
sim.plot()
