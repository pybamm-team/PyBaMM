#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
# load models
options = {
    "particle phases": ("2", "1"),
    "open-circuit potential": (("single", "current sigmoid"), "single"),
}
models = [
    pybamm.lithium_ion.SPM(options),
    pybamm.lithium_ion.SPMe(options),
    pybamm.lithium_ion.DFN(options),
]

parameter_values = pybamm.ParameterValues("Chen2020_composite")

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 4500])
    sims.append(sim)

# plot
pybamm.dynamic_plot(
    sims,
    [
        [
            "Average negative primary particle concentration",
            "Average negative secondary particle concentration",
        ],
        [
            "X-averaged negative electrode primary volumetric "
            "interfacial current density [A.m-3]",
            "X-averaged negative electrode secondary volumetric "
            "interfacial current density [A.m-3]",
            "X-averaged negative electrode volumetric "
            "interfacial current density [A.m-3]",
        ],
        "X-averaged negative electrode primary open-circuit potential [V]",
        "X-averaged negative electrode secondary open-circuit potential [V]",
        "Average positive particle concentration [mol.m-3]",
        "Voltage [V]",
    ],
)
