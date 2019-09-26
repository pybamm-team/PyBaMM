import pybamm
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")

# load (1+1D) SPM model
options = {
    "current collector": "jelly roll",
    "dimensionality": 1,
    "thermal": "set external temperature",
}
model = pybamm.lithium_ion.SPM(options)

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
nbat = 2
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5, var.z: nbat}
# depending on number of points in y-z plane may need to increase recursion depth...
sys.setrecursionlimit(10000)
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)


# define a method which updates statevector
def update_statevector(variables, statevector):
    "takes in a dict of variable name and vector of updated state"
    for name, new_vector in variables.items():
        var_slice = model.variables[name].y_slices
        statevector[var_slice] = new_vector
    return statevector


# define a method to non-dimensionalise a temperature
def non_dim_temperature(temperature):
    "takes in a temperature and returns the non-dimensional version"
    Delta_T = param.process_symbol(model.submodels["thermal"].param.Delta_T).evaluate()
    T_ref = param.process_symbol(model.submodels["thermal"].param.T_ref).evaluate()
    return (temperature - T_ref) / Delta_T


# step model in time
solver = model.default_solver
dt = 0.1  # timestep to take
npts = 20  # number of points to store the solution at during this step
solution1 = solver.step(model, dt, npts=npts)
phi_s_cn_step1 = pybamm.ProcessedVariable(
    model.variables["Negative current collector potential [V]"],
    solution1.t,
    solution1.y,
    mesh=mesh,
)
phi_s_cp_step1 = pybamm.ProcessedVariable(
    model.variables["Positive current collector potential [V]"],
    solution1.t,
    solution1.y,
    mesh=mesh,
)
voltage_step1 = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solution1.t, solution1.y, mesh=mesh
)
current_step1 = pybamm.ProcessedVariable(
    model.variables["Current [A]"], solution1.t, solution1.y, mesh=mesh
)
heating_step1 = pybamm.ProcessedVariable(
    model.variables["X-averaged total heating [A.V.m-3]"],
    solution1.t,
    solution1.y,
    mesh=mesh,
)
particle_step1 = pybamm.ProcessedVariable(
    model.variables["X-averaged positive particle surface concentration [mol.m-3]"],
    solution1.t,
    solution1.y,
    mesh=mesh,
)
temperature_step1 = pybamm.ProcessedVariable(
    model.variables["X-averaged cell temperature [K]"],
    solution1.t,
    solution1.y,
    mesh=mesh,
)

# get the current state and temperature
current_state = solution1.y[:, -1]
temp_ave = current_state[model.variables["X-averaged cell temperature"].y_slices]

# update the temperature
T_ref = param.process_symbol(model.submodels["thermal"].param.T_ref).evaluate()
t_external = np.linspace(T_ref, T_ref + 6.0, nbat)
non_dim_t_external = non_dim_temperature(t_external)
variables = {"X-averaged cell temperature": non_dim_t_external}
new_state = update_statevector(variables, current_state)

# step in time again
# use new state as initial condition. Note: need to to recompute consistent initial
# values for the algebraic part of the model. Since the (dummy) equation for the
# temperature is an ODE, the imposed change in temperature is unaffected by this
# process
solver.y0 = solver.calculate_consistent_initial_conditions(
    solver.rhs, solver.algebraic, new_state
)
solution2 = solver.step(model, dt, npts=npts)
phi_s_cn_step2 = pybamm.ProcessedVariable(
    model.variables["Negative current collector potential [V]"],
    solution2.t,
    solution2.y,
    mesh=mesh,
)
phi_s_cp_step2 = pybamm.ProcessedVariable(
    model.variables["Positive current collector potential [V]"],
    solution2.t,
    solution2.y,
    mesh=mesh,
)
voltage_step2 = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solution2.t, solution2.y, mesh=mesh
)
current_step2 = pybamm.ProcessedVariable(
    model.variables["Current [A]"], solution2.t, solution2.y, mesh=mesh
)
heating_step2 = pybamm.ProcessedVariable(
    model.variables["X-averaged total heating [A.V.m-3]"],
    solution2.t,
    solution2.y,
    mesh=mesh,
)
particle_step2 = pybamm.ProcessedVariable(
    model.variables["X-averaged positive particle surface concentration [mol.m-3]"],
    solution2.t,
    solution2.y,
    mesh=mesh,
)
temperature_step2 = pybamm.ProcessedVariable(
    model.variables["X-averaged cell temperature [K]"],
    solution2.t,
    solution2.y,
    mesh=mesh,
)

# plot
t_sec = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.tau_discharge
).evaluate()
t_hour = t_sec / (3600)
plt.figure()
z = np.linspace(0, 1, nbat)
for bat_id in range(nbat):
    plt.plot(
        solution1.t * t_hour,
        phi_s_cp_step1(solution1.t, z=z)[bat_id, :]
        - phi_s_cn_step1(solution1.t, z=z)[bat_id, :],
        solution2.t * t_hour,
        phi_s_cp_step2(solution2.t, z=z)[bat_id, :]
        - phi_s_cn_step2(solution2.t, z=z)[bat_id, :],
    )
plt.xlabel("t [hrs]")
plt.ylabel("Local voltage [V]")
plt.figure()
plt.plot(
    solution1.t, voltage_step1(solution1.t), solution2.t, voltage_step2(solution2.t)
)
plt.xlabel("t")
plt.ylabel("Voltage [V]")
plt.show()
plt.figure()
plt.plot(
    solution1.t, current_step1(solution1.t), solution2.t, current_step2(solution2.t)
)
plt.xlabel("t")
plt.ylabel("Current [A]")
plt.show()
plt.figure()
z = np.linspace(0, 1, nbat)
for bat_id in range(nbat):
    plt.plot(
        solution1.t * t_hour,
        heating_step1(solution1.t, z=z)[bat_id, :],
        solution2.t * t_hour,
        heating_step2(solution2.t, z=z)[bat_id, :],
    )
plt.xlabel("t [hrs]")
plt.ylabel("X-averaged total heating [A.V.m-3]")
plt.yscale("log")
plt.show()
plt.figure()
for bat_id in range(nbat):
    plt.plot(
        solution1.t * t_hour,
        particle_step1(solution1.t, z=z)[bat_id, :],
        solution2.t * t_hour,
        particle_step2(solution2.t, z=z)[bat_id, :],
    )
plt.xlabel("t [hrs]")
plt.ylabel("X-averaged positive particle surface concentration [mol.m-3]")
plt.show()
plt.figure()
for bat_id in range(nbat):
    plt.plot(
        solution1.t * t_hour,
        temperature_step1(solution1.t, z=z)[bat_id, :],
        solution2.t * t_hour,
        temperature_step2(solution2.t, z=z)[bat_id, :],
    )
plt.xlabel("t [hrs]")
plt.ylabel("X-averaged cell temperature [K]")
plt.show()


def plot_var(var, solution, time=-1):
    variable = model.variables[var]
    len_x = len(mesh.combine_submeshes(*variable.domain))
    len_z = variable.shape[0] // len_x
    entries = np.empty((len_x, len_z, len(solution.t)))

    for idx in range(len(solution.t)):
        t = solution.t[idx]
        y = solution.y[:, idx]
        entries[:, :, idx] = np.reshape(variable.evaluate(t, y), [len_x, len_z])
    plt.figure()
    for bat_id in range(len_x):
        plt.plot(range(len_z), entries[bat_id, :, time].flatten())
    plt.title(var)
    plt.figure()
    plt.imshow(entries[:, :, time])
    plt.title(var)


# plot_var(var="Positive current collector potential", solution=solution1, time=-1)
# plot_var(var="Total heating [A.V.m-3]", solution=solution1, time=-1)
# plot_var(var="Interfacial current density", solution=solution2, time=-1)
# plot_var(var="Negative particle concentration [mol.m-3]", solution=solution2, time=-1)
# plot_var(var="Positive particle concentration [mol.m-3]", solution=solution2, time=-1)

var_names = list(model.variables.keys())
var_names.sort()
