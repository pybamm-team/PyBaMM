#
# Compare models for diffusion within the electrode particles
#
import pybamm

# pybamm.set_logging_level("INFO")
# pybamm.set_logging_level("DEBUG")
# load models
models = [
    pybamm.lithium_ion.SPMe(
        options={"particle": "Fickian diffusion"}, name="Fickian diffusion"
    ),
    pybamm.lithium_ion.SPMe(
        options={"particle": "fast diffusion"}, name="fast diffusion"
    ),
    pybamm.lithium_ion.SPMe(options={"particle": "uniform"}, name="uniform profile"),
    pybamm.lithium_ion.SPMe(
        options={"particle": "quadratic profile"}, name="quadratic profile"
    ),
]

# create and solve simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model)
    sim.solve([0, 3600])
    sims.append(sim)
    print("Particle model: {}".format(model.name))
    print("Solve time: {}s".format(sim.solution.solve_time))

# plot results
pybamm.dynamic_plot(sims)
