#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
# load models
options = {
    "particle phases": ("2", "1"),
    "open circuit potential": (("single", "current sigmoid"), "single"),
}
models = [
    # pybamm.lithium_ion.SPM({"particle phases": ("2", "1")}),
    # pybamm.lithium_ion.SPMe({"particle phases": ("2", "1")}),
    pybamm.lithium_ion.DFN(options),
    pybamm.lithium_ion.BasicDFNComposite(),
]

parameter_values = pybamm.ParameterValues("Chen2020_composite")

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(
    sims,
    [
        [
            "Average negative primary particle concentration",
            "Average negative secondary particle concentration",
        ],
        "X-averaged negative electrode primary interfacial current density",
        "X-averaged negative electrode secondary interfacial current density",
        "X-averaged negative electrode primary open circuit potential [V]",
        "X-averaged negative electrode secondary open circuit potential [V]",
        "X-averaged negative electrode interfacial current density",
        # "Electrolyte concentration",
        "Average positive particle concentration [mol.m-3]",
        # "Current [A]",
        # "Negative electrode potential [V]",
        # "Electrolyte potential [V]",
        # "Positive electrode potential [V]",
        "Terminal voltage [V]",
        # ["Negative primary particle radius", "Negative secondary particle radius"],
    ],
)
