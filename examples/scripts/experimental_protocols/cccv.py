#
# Constant-current constant-voltage charge
#
import pybamm
import matplotlib.pyplot as plt

pybamm.set_logging_level("NOTICE")

yang = pybamm.lithium_ion.Yang2017()
dfn = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Yang2017)

Vmin = 2.5
Vmax = 4.2
experiment = pybamm.Experiment(
    [
        (
            f"Discharge at 1C until {Vmin}V",
            "Rest for 1 hour",
            f"Charge at 1C until {Vmax}V",
            f"Hold at {Vmax}V until C/50",
        )
    ],
    # period="10 seconds",
)

sim1 = pybamm.Simulation(
    yang, experiment=experiment, solver=pybamm.CasadiSolver("fast with events")
)  # , parameter_values=parameter_values)
sol1 = sim1.solve()

sim2 = pybamm.Simulation(
    dfn,
    experiment=experiment,
    solver=pybamm.CasadiSolver("fast with events"),
    parameter_values=parameter_values,
)
sol2 = sim2.solve()


pybamm.dynamic_plot(
    [sol1, sol2],
    [
        "Terminal voltage [V]",
        "Current [A]",
        "X-averaged negative electrode interfacial current density",
        "X-averaged negative electrode SEI interfacial current density",
        "X-averaged negative electrode lithium plating interfacial current density",
        "X-averaged negative electrode SEI film overpotential [V]",
    ],
)
# Plot voltages from the discharge segments only
# fig, ax = plt.subplots()
# for i in range(3):
#     # Extract sub solutions
#     sol = sim.solution.cycles[i]
#     # Extract variables
#     t = sol["Time [h]"].entries
#     V = sol["Terminal voltage [V]"].entries
#     # Plot
#     ax.plot(t - t[0], V, label="Discharge {}".format(i + 1))
#     ax.set_xlabel("Time [h]")
#     ax.set_ylabel("Voltage [V]")
#     ax.set_xlim([0, 10])
# ax.legend(loc="lower left")

# Save time, voltage, current, discharge capacity, temperature, and electrolyte
# concentration to csv and matlab formats
sim.solution.save_data(
    "output.mat",
    [
        "Time [h]",
        "Current [A]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [K]",
        "Electrolyte concentration [mol.m-3]",
    ],
    to_format="matlab",
    short_names={
        "Time [h]": "t",
        "Current [A]": "I",
        "Terminal voltage [V]": "V",
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
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [K]",
    ],
    to_format="csv",
)

# Show all plots
sim.plot()
