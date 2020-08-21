#
# Compare models for diffusion within the electrode particles
#
import pybamm

# load models
models = [
    pybamm.lithium_ion.DFN(
        options={"particle": "Fickian diffusion"}, name="Fickian diffusion"
    ),
    pybamm.lithium_ion.DFN(
        options={"particle": "uniform profile"}, name="uniform profile"
    ),
    pybamm.lithium_ion.DFN(
        options={"particle": "quadratic profile"}, name="quadratic profile"
    ),
    pybamm.lithium_ion.DFN(
        options={"particle": "quartic profile"}, name="quartic profile"
    ),
]

# pick parameter values
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)

# create and solve simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 3600])
    sims.append(sim)
    print("Particle model: {}".format(model.name))
    print("Solve time: {}s".format(sim.solution.solve_time))

# plot results
pybamm.dynamic_plot(sims)
