#
# Constant-current constant-voltage charge
#
import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("NOTICE")

yang = pybamm.lithium_ion.Yang2017()
dfn = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Yang2017)

yang.variables["Left porosity"] = pybamm.boundary_value(
    yang.variables["Negative electrode porosity"], "left"
)
yang.variables["Right porosity"] = pybamm.boundary_value(
    yang.variables["Negative electrode porosity"], "right"
)

Vmin = 2.8
Vmax = 4.2
experiment = pybamm.Experiment(
    [
        (
            f"Discharge at 2C until {Vmin}V",
            "Rest for 5 minutes",
            # f"Charge at 1C until {Vmax}V",
            # f"Hold at {Vmax}V until C/10",
            # "Rest for 5 minutes",
        )
    ]
    * 1,
    # + [
    #     (
    #         f"Discharge at 1C until {Vmin}V",
    #         "Rest for 1 hour",
    #     )
    # ],
    termination="80% capacity"
    # period="10 seconds",
)
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 10, var.r_p: 10}
sim1 = pybamm.Simulation(
    yang,
    experiment=experiment,
    var_pts=var_pts,
    solver=pybamm.CasadiSolver("fast with events", atol=1e-6, rtol=1e-6),
    parameter_values=parameter_values,
)
sol1 = sim1.solve()
print(sol1.summary_variables["n_Li"])
print(np.diff(sol1.summary_variables["n_Li"]))

sim2 = pybamm.Simulation(
    dfn,
    experiment=experiment,
    var_pts=var_pts,
    solver=pybamm.CasadiSolver("fast with events", atol=1e-6, rtol=1e-6),
    parameter_values=parameter_values,
)
# sol2 = sim2.solve()


pybamm.dynamic_plot(
    [sol1],
    [
        # "Terminal voltage [V]",
        # "Current [A]",
        # "X-averaged negative electrode interfacial current density",
        # "X-averaged negative electrode SEI interfacial current density",
        # "X-averaged negative electrode lithium plating interfacial current density",
        # "X-averaged negative electrode SEI film overpotential [V]",
        # "X-averaged negative electrode reaction overpotential [V]",
        # "X-averaged negative electrode lithium plating reaction overpotential [V]",
        # "X-averaged negative electrode porosity",
        # ["Left porosity", "Right porosity"],
        # "Loss of Lithium Inventory [%]",
        # "Negative electrode SOC",
        # "Positive electrode SOC",
        "Sum of electrolyte reaction source terms",
        # "Total negative electrode interfacial current density variable",
        "X-averaged electrolyte concentration",
        "Total lithium in electrolyte [mol]",
        # "Negative electrode porosity",
    ],
    variable_limits="tight",
)
# # Plot voltages from the discharge segments only
# # fig, ax = plt.subplots()
# # for i in range(3):
# #     # Extract sub solutions
# #     sol = sim.solution.cycles[i]
# #     # Extract variables
# #     t = sol["Time [h]"].entries
# #     V = sol["Terminal voltage [V]"].entries
# #     # Plot
# #     ax.plot(t - t[0], V, label="Discharge {}".format(i + 1))
# #     ax.set_xlabel("Time [h]")
# #     ax.set_ylabel("Voltage [V]")
# #     ax.set_xlim([0, 10])
# # ax.legend(loc="lower left")

# # Save time, voltage, current, discharge capacity, temperature, and electrolyte
# # concentration to csv and matlab formats
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

# # Show all plots
# sim.plot()
