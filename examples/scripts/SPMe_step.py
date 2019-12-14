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
t_eval = np.linspace(0, 0.2, 100)
solver = model.default_solver
solution = solver.solve(model, t_eval)

# step model
dt = 0.05
time = 0
end_time = solution.t[-1]
step_solver = model.default_solver
step_solution = None
while time < end_time:
    current_step_sol = step_solver.step(model, dt=dt, npts=10)
    if not step_solution:
        # create solution object on first step
        step_solution = current_step_sol
    else:
        # append solution from the current step to step_solution
        step_solution.append(current_step_sol)
    time += dt

# plot
<<<<<<< HEAD
voltage = solution["Terminal voltage [V]"]
step_voltage = step_solution["Terminal voltage [V]"]
=======
voltage = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solution.t, solution.y, mesh=mesh
)
step_voltage = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], step_solution.t, step_solution.y, mesh=mesh
)
>>>>>>> issue-492-potentiostatic
plt.plot(solution.t, voltage(solution.t), "b-", label="SPMe (continuous solve)")
plt.plot(
    step_solution.t, step_voltage(step_solution.t), "ro", label="SPMe (steppped solve)"
)
plt.xlabel(r"$t$")
plt.ylabel("Terminal voltage [V]")
plt.legend()
plt.show()
