from pybamm.parameters import Parameters
from pybamm.mesh import Mesh
from pybamm.models.model_class import Model
from pybamm.simulation import Simulation
from pybamm.solver import Solver

# Create parameters and mesh
param = Parameters()
tsteps = 100
tend = 1
target_npts = 10
mesh = Mesh(param, target_npts, tsteps=tsteps, tend=tend)

# Choose the model
model = Model("Electrolyte diffusion")

# Create a simulation
simulation = Simulation(model, param, mesh, name="Electrolyte diffusion")

# Create a solver
solver = Solver(integrator="BDF", spatial_discretisation="Finite Volumes")

# Run the simulation
simulation.run(solver)

# Save
# simulation.save(filename)

# Plot
# simulation.plot()
