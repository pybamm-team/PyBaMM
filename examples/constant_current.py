from pybat_lead_acid.parameters import Parameters
from pybat_lead_acid.mesh import Mesh
from pybat_lead_acid.models.model_class import Model
from pybat_lead_acid.simulation import Simulation
from pybat_lead_acid.solver import Solver

import numpy as np

# Create parameters and mesh
param = Parameters()
tsteps = 100
tend = 1
target_npts = 10
mesh = Mesh(param, target_npts, tsteps=tsteps, tend=tend)

# Choose the model
model = Model("Simple Diffusion")

# Create a simulation
simulation = Simulation(model, param, mesh, name="Simple Diffusion")

# Create a solver
solver = Solver(integrator="BDF", spatial_discretisation="Finite Volumes")

# Run the simulation
simulation.run(solver)
import ipdb; ipdb.set_trace()
# Save
simulation.save(filename)

# Plot
simulation.plot()
