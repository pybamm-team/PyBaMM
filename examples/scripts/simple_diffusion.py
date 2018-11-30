#
# Run the reaction-diffusion model
#
import pybamm

model = pybamm.ReactionDiffusionModel()
sim = pybamm.Simulation(model)
sim.run()
