#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
# load models
models = [
    # pybamm.lithium_ion.SPMe(),
    pybamm.lithium_ion.SPM(
        {"particle": "uniform profile", "surface form": "algebraic"}
    ),
    pybamm.lithium_ion.SPM({"surface form": "algebraic"}),
    # pybamm.lithium_ion.SPMe(options),
    # pybamm.lithium_ion.SPMe({"thermal": "lumped"}),
    # pybamm.lithium_ion.DFN(options),
    # pybamm.lithium_ion.NewmanTobias({"particle": "uniform profile"}),
    # pybamm.lithium_ion.NewmanTobias(
    #     {"particle": "uniform profile", "surface form": "algebraic"}
    # ),
    # pybamm.lithium_ion.NewmanTobias({"surface form": "differential"}),
]


parameter_values = pybamm.ParameterValues("Marquis2019")
parameter_values["Electrolyte conductivity [S.m-1]"] = 1
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(
    sims,
    [
        "Total lithium in particles [mol]",
        "X-averaged negative electrode volumetric interfacial current density [A.m-3]",
        "X-averaged positive electrode volumetric interfacial current density [A.m-3]",
        # "X-averaged negative particle concentration [mol.m-3]",
        # "X-averaged positive particle concentration [mol.m-3]",
        # "Negative electrode filling fraction",
        # "Positive electrode filling fraction",
        # "Negative electrode interfacial current density [A.m-2]",
        # "Current collector current density [A.m-2]",
        # "Electrolyte current density [A.m-2]",
    ],
)
