#
# Constant-current constant-voltage charge
#
import pybamm
import matplotlib.pyplot as plt

pybamm.set_logging_level("NOTICE")

Vmin = 3.0
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
    termination="80% capacity",
)
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Mohtat2020)
parameter_values.update(
    {
        "Exchange-current density for plating [A.m-2]": 0.001,
        "Initial plated lithium concentration [mol.m-3]": 0,
        "Lithium metal partial molar volume [m3.mol-1]": 1.3e-5,
        "SEI kinetic rate constant [m.s-1]": 1e-15,
        "SEI resistivity [Ohm.m]": 0,
    },
)
spm = pybamm.lithium_ion.SPM(
    {
        "SEI": "ec reaction limited",
        "SEI film resistance": "none",
        "lithium plating": "irreversible",
    }
)
sim = pybamm.Simulation(
    spm,
    experiment=experiment,
    parameter_values=parameter_values,
    solver=pybamm.CasadiSolver("safe"),
)
starting_sol = pybamm.load("spm_sol_100.sav")
sim.solve(starting_solution=starting_sol)

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
