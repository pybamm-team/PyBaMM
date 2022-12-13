#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
options = {"thermal": "x-full"}
# load models
models = [
    pybamm.lithium_ion.SPMe(),
    # pybamm.lithium_ion.SPM(options),
    pybamm.lithium_ion.SPMe(options),
    # pybamm.lithium_ion.DFN(options),
    # pybamm.lithium_ion.NewmanTobias(),
]


parameter_values = pybamm.ParameterValues("Marquis2019")

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
        # "X-averaged negative particle concentration [mol.m-3]",
        # "X-averaged positive particle concentration [mol.m-3]",
        # "Negative electrode filling fraction",
        # "Positive electrode filling fraction",
        # "Negative electrode interfacial current density [A.m-2]",
        # "Current collector current density [A.m-2]",
        # "Electrolyte current density [A.m-2]",
    ],
)
