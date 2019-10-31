#
# Example of 1+1D SPM where the temperature can be set by the user
#

import pybamm
import numpy as np
import matplotlib.pyplot as plt
import sys

# set logging level
pybamm.set_logging_level("INFO")

# load (1+1D) SPM model
options = {
    "current collector": "potential pair",
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
nbat = 5
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


# step model in time and process variables for later plotting
solver = model.default_solver
dt = 0.1  # timestep to take
npts = 20  # number of points to store the solution at during this step
solution1 = solver.step(model, dt, npts=npts)
# create dict of variables to post process
output_variables = [
    "Negative current collector potential [V]",
    "Positive current collector potential [V]",
    "Current [A]",
    "X-averaged total heating [A.V.m-3]",
    "X-averaged positive particle surface concentration [mol.m-3]",
    "X-averaged cell temperature [K]",
]
output_variables_dict = {}
for var in output_variables:
    output_variables_dict[var] = model.variables[var]
processed_vars_step1 = pybamm.post_process_variables(
    output_variables_dict, solution1.t, solution1.y, mesh
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

# step in time again and process variables for later plotting
# use new state as initial condition. Note: need to to recompute consistent initial
# values for the algebraic part of the model. Since the (dummy) equation for the
# temperature is an ODE, the imposed change in temperature is unaffected by this
# process
solver.y0 = solver.calculate_consistent_initial_conditions(
    solver.rhs, solver.algebraic, new_state
)
solution2 = solver.step(model, dt, npts=npts)
processed_vars_step2 = pybamm.post_process_variables(
    output_variables_dict, solution2.t, solution2.y, mesh
)

# plots
t_sec = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_hour = t_sec / (3600)
z = np.linspace(0, 1, nbat)

# local voltage
plt.figure()
for bat_id in range(nbat):
    plt.plot(
        solution1.t * t_hour,
        processed_vars_step1["Positive current collector potential [V]"](
            solution1.t, z=z
        )[bat_id, :]
        - processed_vars_step1["Negative current collector potential [V]"](
            solution1.t, z=z
        )[bat_id, :],
        solution2.t * t_hour,
        processed_vars_step2["Positive current collector potential [V]"](
            solution2.t, z=z
        )[bat_id, :]
        - processed_vars_step1["Negative current collector potential [V]"](
            solution2.t, z=z
        )[bat_id, :],
    )
plt.xlabel("t [hrs]")
plt.ylabel("Local voltage [V]")

# applied current
plt.figure()
plt.plot(
    solution1.t,
    processed_vars_step1["Current [A]"](solution1.t),
    solution2.t,
    processed_vars_step2["Current [A]"](solution2.t),
)
plt.xlabel("t")
plt.ylabel("Current [A]")

# local heating
plt.figure()
z = np.linspace(0, 1, nbat)
for bat_id in range(nbat):
    plt.plot(
        solution1.t * t_hour,
        processed_vars_step1["X-averaged total heating [A.V.m-3]"](solution1.t, z=z)[
            bat_id, :
        ],
        solution2.t * t_hour,
        processed_vars_step2["X-averaged total heating [A.V.m-3]"](solution2.t, z=z)[
            bat_id, :
        ],
    )
plt.xlabel("t [hrs]")
plt.ylabel("X-averaged total heating [A.V.m-3]")
plt.yscale("log")

# local concentration
plt.figure()
for bat_id in range(nbat):
    plt.plot(
        solution1.t * t_hour,
        processed_vars_step1[
            "X-averaged positive particle surface concentration [mol.m-3]"
        ](solution1.t, z=z)[bat_id, :],
        solution2.t * t_hour,
        processed_vars_step2[
            "X-averaged positive particle surface concentration [mol.m-3]"
        ](solution2.t, z=z)[bat_id, :],
    )
plt.xlabel("t [hrs]")
plt.ylabel("X-averaged positive particle surface concentration [mol.m-3]")

# local temperature
plt.figure()
for bat_id in range(nbat):
    plt.plot(
        solution1.t * t_hour,
        processed_vars_step1["X-averaged cell temperature [K]"](solution1.t, z=z)[
            bat_id, :
        ],
        solution2.t * t_hour,
        processed_vars_step2["X-averaged cell temperature [K]"](solution2.t, z=z)[
            bat_id, :
        ],
    )
plt.xlabel("t [hrs]")
plt.ylabel("X-averaged cell temperature [K]")

plt.show()
