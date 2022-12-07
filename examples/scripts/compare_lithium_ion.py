#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
# load models
models = [
    # pybamm.lithium_ion.SPM(),
    pybamm.lithium_ion.SPM(
        {
            "lithium plating": "irreversible",
            "lithium plating porosity change": "true",
        }
    ),
    # pybamm.lithium_ion.SPMe(options),
    # pybamm.lithium_ion.DFN(options),
    # pybamm.lithium_ion.NewmanTobias(),
]

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=pybamm.ParameterValues("OKane2022"))
    sim.solve([0, 4000])
    sims.append(sim)

# plot
pybamm.dynamic_plot(
    sims,
    [
        "Total lithium in electrolyte [mol]",
    ],
)
