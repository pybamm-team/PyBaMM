import pybamm

# Create parameters and mesh
param = pybamm.Parameters()
tsteps = 100
tend = 1
target_npts = 10
mesh = pybamm.Mesh(param, target_npts, tsteps=tsteps, tend=tend)

# Choose the model
model = pybamm.Model("Electrolyte diffusion")

# Create a simulation
simulation = pybamm.Simulation(
    model, param, mesh, name="Electrolyte diffusion"
)

# Create a solver
solver = pybamm.Solver(
    integrator="BDF", spatial_discretisation="Finite Volumes"
)

# Run the simulation
simulation.run(solver)

# Save
# simulation.save(filename)

# Plot
# simulation.plot()
