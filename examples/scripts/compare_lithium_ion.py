#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
options = {"loss of active material": ("none", "stress-driven")}
# options = {"particle mechanics": ("none", "swelling and cracking")}
parameter_values = pybamm.ParameterValues("Ai2020")

# load models
models = [
    # pybamm.lithium_ion.SPM(),
    pybamm.lithium_ion.SPM(options),
    # pybamm.lithium_ion.SPMe(options),
    # pybamm.lithium_ion.DFN(options),
    # pybamm.lithium_ion.NewmanTobias(),
]

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 4000])
    sims.append(sim)

# plot
pybamm.dynamic_plot(
    sims,
    ["Total lithium in system [mol]"],
)
