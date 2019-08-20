import pybamm
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.close('all')
# set logging level
pybamm.set_logging_level("INFO")

# load (1+1D) SPM model
options = {"current collector": "set external potential",
           "dimensionality": 1,
           "thermal": "set external temperature"}
model = pybamm.lithium_ion.SPM(options)

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
nbat = 10
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5, var.z: nbat}
# depnding on number of points in y-z plane may need to increase recursion depth...
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
    return statevector[:, np.newaxis]  # should be column vector


# define a method which takes a dimensional potential [V] and converts to the
# dimensionless potential used in pybamm
pot_scale = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.potential_scale
).evaluate()  # potential scaled on thermal voltage
pot_ref = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.U_p_ref
    - pybamm.standard_parameters_lithium_ion.U_n_ref
).evaluate()  # positive potential measured with respect to reference OCV


def non_dim_potential(phi_dim, domain):
    if domain == "negative":
        phi = phi_dim / pot_scale
    elif domain == "positive":
        phi = (phi_dim - pot_ref) / pot_scale
    return phi


# solve model -- replace with step
t_eval1 = np.linspace(0, 0.1, 20)
solution1 = model.default_solver.solve(model, t_eval1)
voltage_step1 = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solution1.t, solution1.y, mesh=mesh
)
current_step1 = pybamm.ProcessedVariable(
    model.variables["Current [A]"], solution1.t, solution1.y, mesh=mesh
)
heating_step1 = pybamm.ProcessedVariable(
    model.variables["X-averaged total heating [A.V.m-3]"], solution1.t, solution1.y, mesh=mesh
)
particle_step1 = pybamm.ProcessedVariable(
    model.variables["X-averaged positive particle surface concentration [mol.m-3]"], solution1.t, solution1.y, mesh=mesh
)


current_state = solution1.y[:, -1]

# update potentials (e.g. zero volts on neg. current collector, 3.3 volts on pos.)
#phi_s_cn_dim_new = np.zeros(var_pts[var.z])
sf_cn = 1.0
phi_s_cn_dim_new = current_state[model.variables["Negative current collector potential"].y_slices] * sf_cn

#phi_s_cp_dim_new = 3.3 * np.ones(var_pts[var.z]) - 0.05 * np.linspace(0, 1, var_pts[var.z])
#phi_s_cp_dim_new = 3.3 * np.ones(var_pts[var.z])
sf_cp = 1e-2
phi_s_cp_dim_new = current_state[model.variables["Positive current collector potential"].y_slices] - sf_cp * np.linspace(0, 1, var_pts[var.z])
#variables = {
#    "Negative current collector potential": non_dim_potential(
#        phi_s_cn_dim_new, "negative"
#    ),
#    "Positive current collector potential": non_dim_potential(
#        phi_s_cp_dim_new, "positive"
#    ),
#}

variables = {
    "Negative current collector potential": phi_s_cn_dim_new,
    "Positive current collector potential": phi_s_cp_dim_new,
}

new_state = update_statevector(variables, current_state)

# solve again -- replace with step
# use new state as initial condition
model.concatenated_initial_conditions = new_state
#model.concatenated_initial_conditions = current_state[:, np.newaxis]
t_eval2 = np.linspace(0.1, 0.2, 20)
solution2 = model.default_solver.solve(model, t_eval2)
voltage_step2 = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solution2.t, solution2.y, mesh=mesh
)
current_step2 = pybamm.ProcessedVariable(
    model.variables["Current [A]"], solution2.t, solution2.y, mesh=mesh
)
heating_step2 = pybamm.ProcessedVariable(
    model.variables["X-averaged total heating [A.V.m-3]"], solution2.t, solution2.y, mesh=mesh
)
particle_step2 = pybamm.ProcessedVariable(
    model.variables["X-averaged positive particle surface concentration [mol.m-3]"], solution2.t, solution2.y, mesh=mesh
)
# plot
#plt.figure()
#plt.plot(t_eval1, voltage_step1(t_eval1), t_eval2, voltage_step2(t_eval2))
#plt.xlabel('t')
#plt.ylabel('Voltage [V]')
#plt.show()
#plt.figure()
#plt.plot(t_eval1, current_step1(t_eval1), t_eval2, current_step2(t_eval2))
#plt.xlabel('t')
#plt.ylabel('Current [A]')
#plt.show()
#plt.figure()
z = np.linspace(0, 1, 10)
for bat_id in range(nbat):
    plt.plot(t_eval1, heating_step1(t_eval1, z=z)[bat_id, :], t_eval2, heating_step2(t_eval2, z=z)[bat_id, :])
plt.xlabel('t')
plt.ylabel('X-averaged total heating [A.V.m-3]')
plt.yscale('log')
plt.show()
plt.figure()
for bat_id in range(nbat):
    plt.plot(t_eval1, particle_step1(t_eval1,z=z)[bat_id, :], t_eval2, particle_step2(t_eval2,z=z)[bat_id, :])
plt.xlabel('t')
plt.ylabel('X-averaged positive particle surface concentration [mol.m-3]')
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

#plot_var(var="Positive current collector potential", solution=solution1, time=-1)
#plot_var(var="Total heating [A.V.m-3]", solution=solution1, time=-1)
#plot_var(var="Interfacial current density", solution=solution2, time=-1)
#plot_var(var="Negative particle concentration [mol.m-3]", solution=solution2, time=-1)
#plot_var(var="Positive particle concentration [mol.m-3]", solution=solution2, time=-1)

var_names = list(model.variables.keys())
var_names.sort()
