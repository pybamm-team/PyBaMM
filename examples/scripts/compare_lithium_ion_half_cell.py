#
# Compare half-cell lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    # pybamm.lithium_ion.SPM({"working electrode": "positive"}),
    # pybamm.lithium_ion.SPMe({"working electrode": "positive"}),
    pybamm.lithium_ion.DFN(
        {
            "working electrode": "positive",
            "interface utilisation": ("current-driven", "full"),
        }
    ),
    pybamm.lithium_ion.DFN({"working electrode": "positive"}),
]

chemistry = pybamm.parameter_sets.Xu2019
param = pybamm.ParameterValues(chemistry=chemistry)
param.update(
    {
        "Negative electrode current-driven interface utilisation factor [m3.mol-1]": -1,
        "Initial negative electrode interface utilisation": 1,
    },
    check_already_exists=False,
)

experiment = pybamm.Experiment(
    ["Discharge at 2C until 3.5V", "Charge at 2C until 4.2V"]
)

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(
    sims, ["Lithium metal interface utilisation", "Terminal voltage [V]"]
)
