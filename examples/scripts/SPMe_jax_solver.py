"""
Example of Jax solver with the SPMe model. Use the `pybamm_install_jax`
command to automatically download and install jax and jaxlib on your system.
"""

import matplotlib.pylab as plt
import pybamm

# Load the SPMe model with Jax support
model = pybamm.lithium_ion.SPMe(options={'timescale': 1.0})
model.convert_to_format = 'jax'
model.events = []

# Create geometry
geometry = model.default_geometry

# Load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# Set the mesh and discretize the model
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
discrete = pybamm.Discretisation(mesh, model.default_spatial_methods)
discrete.process_model(model)

# Use Jax to solve the model and get solution object
solver = pybamm.JaxSolver()
solution = solver.solve(model, [0, 3600])

# Plot results
_, ax = plt.subplots()
ax.plot(solution['Time [s]'].entries, solution['Terminal voltage [V]'].entries)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Voltage [V]')

plt.show()
