#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
# load models
models = [
    # pybamm.lithium_ion.SPMe(),
    # pybamm.lithium_ion.SPM(
    #     {
    #         # "particle": "uniform profile",
    #         # "surface form": "algebraic",
    #         "intercalation kinetics": "linear",
    #     }
    # ),
    # pybamm.lithium_ion.SPM({"surface form": "algebraic"}),
    # pybamm.lithium_ion.SPMe(options),
    # pybamm.lithium_ion.SPMe({"thermal": "lumped"}),
    pybamm.lithium_ion.DFN(),
    pybamm.lithium_ion.DFN({"surface form": "algebraic"}),
    pybamm.lithium_ion.NewmanTobias({"surface form": "algebraic"}),
    pybamm.lithium_ion.NewmanTobias(),
    # pybamm.lithium_ion.NewmanTobias({"surface form": "differential"}),
]


parameter_values = pybamm.ParameterValues("Marquis2019")
parameter_values["Negative electrode conductivity [S.m-1]"] = 100000
parameter_values["Positive electrode conductivity [S.m-1]"] = 100000
parameter_values["Electrolyte conductivity [S.m-1]"] = 1
parameter_values["Electrolyte diffusivity [m2.s-1]"] = 1e-9
sims = []
for model in models:
    sim = pybamm.Simulation(
        model,
        parameter_values=parameter_values,
        solver=pybamm.CasadiSolver(),
    )
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(
    sims,
    [
        "Total lithium in particles [mol]",
        "Total lithium in electrolyte [mol]",
        "X-averaged negative electrode volumetric interfacial current density [A.m-3]",
        "X-averaged positive electrode volumetric interfacial current density [A.m-3]",
        # "Electrolyte concentration [mol.m-3]",
        # "X-averaged negative particle concentration [mol.m-3]",
        # "X-averaged positive particle concentration [mol.m-3]",
        # "Negative electrode filling fraction",
        # "Positive electrode filling fraction",
        # "Negative electrode interfacial current density [A.m-2]",
        # "Current collector current density [A.m-2]",
        # "Electrolyte current density [A.m-2]",
        # "Negative electrolyte current density [A.m-2]",
        # "Positive electrolyte current density [A.m-2]",
    ],
)


# F = models[0].param.F.value
# R_n = parameter_values.evaluate(models[0].param.n.prim.R_typ)
# R_p = parameter_values.evaluate(models[0].param.p.prim.R_typ)
# L_n = parameter_values.evaluate(models[0].param.n.L)
# L_p = parameter_values.evaluate(models[0].param.p.L)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1, 3)

# for sim in sims:
#     j_n = sim.solution[
#         "X-averaged negative electrode volumetric interfacial current density [A.m-3]"
#     ].data
#     i_cc = sim.solution["Current collector current density [A.m-2]"].data
#     j_n_exact = i_cc / L_n
#     j_p = sim.solution[
#         "X-averaged positive electrode volumetric interfacial current density [A.m-3]"
#     ].data
#     j_p_exact = -i_cc / L_p
#     t = sim.solution["Time [s]"].data

#     from scipy.integrate import cumulative_trapezoid

#     int_j_n = cumulative_trapezoid(j_n, t, initial=0) * 3 / F / R_n
#     int_j_p = cumulative_trapezoid(j_p, t, initial=0) * 3 / F / R_p
#     int_j_n_exact = cumulative_trapezoid(j_n_exact, t, initial=0) * 3 / F / R_n
#     int_j_p_exact = cumulative_trapezoid(j_p_exact, t, initial=0) * 3 / F / R_p

#     ax[0].plot(t, (int_j_n + int_j_p), label=sim.model.name)
#     ax[1].plot(t, j_n)
#     ax[2].plot(t, j_p)
#     ax[0].plot(
#         t,
#         (int_j_n_exact + int_j_p_exact),
#         label=sim.model.name + "exact",
#         linestyle="--",
#     )
#     ax[1].plot(t, j_n_exact, linestyle="--")
#     ax[2].plot(t, j_p_exact, linestyle="--")
# fig.legend()
# fig.tight_layout()
# plt.show()
