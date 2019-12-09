#
# Simulate discharge followed by rest
#
import pybamm
import numpy as np
import matplotlib.pyplot as plt

# load model
pybamm.set_logging_level("INFO")
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

# solve model during discharge stage (1 hour)
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_end = 3600 / tau.evaluate(0)
t_eval1 = np.linspace(0, t_end, 120)
solution1 = model.default_solver.solve(model, t_eval1)

# process variables for later plotting
time1 = pybamm.ProcessedVariable(model.variables["Time [h]"], solution1.t, solution1.y)
voltage1 = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solution1.t, solution1.y, mesh=mesh
)
current1 = pybamm.ProcessedVariable(
    model.variables["Current [A]"], solution1.t, solution1.y, mesh=mesh
)

# solve again with zero current, using last step of solution1 as initial conditions
# update the current to be zero
param["Current function"] = "[zero]"
param.update_model(model, disc)
# Note: need to update model.concatenated_initial_conditions *after* update_model,
# as update_model updates model.concatenated_initial_conditions, by concatenting
# the (unmodified) initial conditions for each variable
model.concatenated_initial_conditions = solution1.y[:, -1][:, np.newaxis]

# simulate 1 hour of rest
t_start = solution1.t[-1]
t_end = t_start + 3600 / tau.evaluate(0)
t_eval2 = np.linspace(t_start, t_end, 120)
solution2 = model.default_solver.solve(model, t_eval2)

# process variables for later plotting
time2 = pybamm.ProcessedVariable(model.variables["Time [h]"], solution2.t, solution2.y)
voltage2 = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solution2.t, solution2.y, mesh=mesh
)
current2 = pybamm.ProcessedVariable(
    model.variables["Current [A]"], solution2.t, solution2.y, mesh=mesh
)

# plot
plt.subplot(121)
plt.plot(time1(t_eval1), voltage1(t_eval1), time2(t_eval2), voltage2(t_eval2))
plt.xlabel("Time [h]")
plt.ylabel("Voltage [V]")
plt.subplot(122)
z = np.linspace(0, 1, 10)
plt.plot(time1(t_eval1), current1(t_eval1), time2(t_eval2), current2(t_eval2))
plt.xlabel("Time [h]")
plt.ylabel("Current [A]")
plt.show()
