#
# Example to compare solving for all times against stepping individually
#
import pybamm
import numpy as np
import matplotlib.pyplot as plt

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
t_eval = np.linspace(0, 3600, 100)
solver = pybamm.CasadiSolver()
solution = solver.solve(model, t_eval)

# step model
dt = 500
time = 0
end_time = solution.t[-1]
step_solver = pybamm.CasadiSolver()
step_solution = None
while time < end_time:
    step_solution = step_solver.step(step_solution, model, dt=dt, npts=10)
    time += dt

# plot
time_in_seconds = solution["Time [s]"].entries
step_time_in_seconds = step_solution["Time [s]"].entries
voltage = solution["Voltage [V]"].entries
step_voltage = step_solution["Voltage [V]"].entries
plt.plot(time_in_seconds, voltage, "b-", label="SPMe (continuous solve)")
plt.plot(step_time_in_seconds, step_voltage, "ro", label="SPMe (stepped solve)")
plt.xlabel(r"$t$")
plt.ylabel("Voltage [V]")
plt.legend()
plt.show()
