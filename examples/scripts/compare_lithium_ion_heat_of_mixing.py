#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
# load models
models = [
    pybamm.lithium_ion.DFN(
        {"heat of mixing": "true", "thermal": "x-lumped"}, name="hom"
    ),
    pybamm.lithium_ion.DFN({"thermal": "x-lumped"}, name="nhom"),
]

parameter_values = pybamm.ParameterValues("Chen2020")

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 4500])
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)
