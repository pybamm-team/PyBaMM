#
# Constant-current constant-voltage charge
#
import pybamm
import matplotlib.pyplot as plt

pybamm.set_logging_level("NOTICE")
experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/5 for 10 hours or until 3.3 V",
            "Rest for 1 hour",
            "Charge at 1 A until 4.1 V",
            "Hold at 4.1 V until 10 mA",
            "Rest for 1 hour",
        ),
    ]
    * 3
)
model = pybamm.lithium_ion.SPM()
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}

sim = pybamm.Simulation(
    model,
    var_pts=var_pts,
    experiment=experiment,
    solver=pybamm.CasadiSolver("fast with events"),  # , rtol=1e-3, atol=1e-3),
)
sim.solve()

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
# sim.solution.save_data(
#     "output.mat",
#     [
#         "Time [h]",
#         "Current [A]",
#         "Terminal voltage [V]",
#         "Discharge capacity [A.h]",
#         "X-averaged cell temperature [K]",
#         "Electrolyte concentration [mol.m-3]",
#     ],
#     to_format="matlab",
#     short_names={
#         "Time [h]": "t",
#         "Current [A]": "I",
#         "Terminal voltage [V]": "V",
#         "Discharge capacity [A.h]": "Q",
#         "X-averaged cell temperature [K]": "T",
#         "Electrolyte concentration [mol.m-3]": "c_e",
#     },
# )
# # We can only save 0D variables to csv
# sim.solution.save_data(
#     "output.csv",
#     [
#         "Time [h]",
#         "Current [A]",
#         "Terminal voltage [V]",
#         "Discharge capacity [A.h]",
#         "X-averaged cell temperature [K]",
#     ],
#     to_format="csv",
# )

# Show all plots
sim.plot(
    [
        "Negative particle surface concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "Positive particle surface concentration [mol.m-3]",
        "Current [A]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        ["Terminal voltage [V]", "Measured open circuit voltage [V]"],
    ]
)
