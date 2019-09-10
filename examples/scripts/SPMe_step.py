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
step_solutions = []
while time < end_time:
    step_sol = step_solver.step(model, dt=dt, npts=10)
    step_solutions.append(step_sol)
    time += dt

# plot
voltage = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"],
    solution.t,
    solution.y,
    mesh=mesh,
)
plt.plot(solution.t, voltage(solution.t), 'b-', label="SPMe")
for i, sol in enumerate(step_solutions):
    voltage = pybamm.ProcessedVariable(
        model.variables["Terminal voltage [V]"],
        sol.t,
        sol.y,
        mesh=mesh,
    )
    plt.plot(sol.t, voltage(sol.t), 'o', label="SPMe Step {}".format(i))
plt.xlabel(r"$t$")
plt.ylabel("Terminal voltage [V]")
plt.legend()
plt.show()
