#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
options = {"loss of active material": ("none", "stress-driven")}
# options = {"particle mechanics": ("none", "swelling and cracking")}
# parameter_values = pybamm.ParameterValues("Ai2020")
parameter_values = pybamm.ParameterValues("Marquis2019")
parameter_values.update(
    {
        "Initial negative electrode interface utilisation": 0.9,
        "Initial positive electrode interface utilisation": 0.8,
        "Negative electrode current-driven interface utilisation factor "
        "[m3.mol-1]": -1e-5,
        "Positive electrode current-driven interface utilisation factor "
        "[m3.mol-1]": 1e-5,
    },
    check_already_exists=False,
)
# load models
models = [
    pybamm.lithium_ion.SPM({"interface utilisation": "current-driven"}),
    # pybamm.lithium_ion.SPMe(),
    # pybamm.lithium_ion.DFN({"loss of active material": "stress-driven"}),
    # pybamm.lithium_ion.NewmanTobias(),
]

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 3600])
    sims.append(
        sim,
    )

# plot
pybamm.dynamic_plot(
    sims,
    [
        "Negative electrode interface utilisation variable",
        "Positive electrode interface utilisation variable",
    ],
)
