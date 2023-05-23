#
# Compare DFN model with and without heat of mixing
#
import pybamm

pybamm.set_logging_level("INFO")
# load models
models = [
    pybamm.lithium_ion.SPMe(
        {"heat of mixing": "true", "thermal": "lumped"}, name="with heat of mixing"
    ),
    pybamm.lithium_ion.DFN({"thermal": "lumped"}, name="without heat of mixing"),
]

parameter_values = pybamm.ParameterValues("Chen2020")

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(
    sims,
    output_variables=[
        "X-averaged cell temperature [K]",
        "X-averaged heat of mixing [W.m-3]",
        "X-averaged total heating [W.m-3]",
        "Voltage [V]",
        "Current [A]",
    ],
)
