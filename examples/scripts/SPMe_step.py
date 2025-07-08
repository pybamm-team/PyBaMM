#
# Example to compare solving for all times against stepping individually
#
import matplotlib.pyplot as plt
import numpy as np

import pybamm

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.SPMe()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = [0, 3700]
solver = pybamm.IDAKLUSolver()
solution = solver.solve(model, t_eval)

# step model
dt = 500
# Set a t_interp to only save the solution at the end of the st
t_interp = [0, dt]
time = 0
end_time = solution.t[-1]
step_solver = pybamm.IDAKLUSolver()
step_solution = None
while time < end_time:
    step_solution = step_solver.step(step_solution, model, dt=dt, t_interp=t_interp)
    time += dt

# plot
t_continuous = solution.t
time_in_seconds = np.linspace(t_continuous[0], t_continuous[-1], 1000)
voltage = solution["Voltage [V]"](time_in_seconds)

step_time_in_seconds = step_solution.t
step_voltage = step_solution["Voltage [V]"].data

plt.plot(time_in_seconds, voltage, "b-", label="SPMe (continuous solve)")
plt.plot(step_time_in_seconds, step_voltage, "ro", label="SPMe (stepped solve)")
plt.xlabel(r"$t$")
plt.ylabel("Voltage [V]")
plt.legend()
plt.show()
