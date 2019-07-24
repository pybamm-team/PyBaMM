import pybamm
import numpy as np
import matplotlib.pyplot as plt
pybamm.set_logging_level("INFO")
ics = {
"Initial concentration in electrolyte [mol.m-3]": 1000.0,
"Initial concentration in negative electrode [mol.m-3]": 19986.609595075,
"Initial concentration in positive electrode [mol.m-3]": 30730.755438556498,
"Initial temperature [K]": 298.15,
}
timestep = 0.1
n_steps = 5
# load model
options = {"thermal": "lumped"}
model = pybamm.lithium_ion.SPMe(options)
# create geometry
geometry = model.default_geometry
# load parameter values and process model and geometry
param = model.default_parameter_values
# Change some parameters
param["Typical current [A]"] = 1.0
param.update(ics)

param.process_model(model)
param.process_geometry(geometry)
# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
# solve model
# Hours
f = 1 / 3600
t_eval = np.arange(n_steps)*timestep*f

# model.use_jacobian = False
# model.use_simplify = False
solver = model.default_solver
#step_sol = solver.solve(model, t_eval)
amalg = []
for i in range(len(t_eval)-1):
    step_sol = solver.step(model, dt=timestep*f)
    amalg.append(step_sol)
cont_sol = model.default_solver.solve_all(model, t_eval)

state_variables = [
    "Electrolyte concentration [mol.m-3]",
    "Negative particle concentration [mol.m-3]",
    "Positive particle concentration [mol.m-3]",
    "Total heating [A.V.m-3]",
]
output = {}
x = model.variables["x"].evaluate().flatten()
x_p = model.variables["x_p"].evaluate().flatten()
x_n = model.variables["x_n"].evaluate().flatten()
r_p = model.variables["r_p"].evaluate().flatten()
r_n = model.variables["r_p"].evaluate().flatten()

print('Solution match', np.allclose(step_sol.y[:, -1], cont_sol.y[:, -1]))

# for i, iv in enumerate(state_variables):

f_step = pybamm.ProcessedVariable(
     model.variables["Electrolyte concentration [mol.m-3]"], step_sol.t, step_sol.y, mesh=mesh
)
f_cont = pybamm.ProcessedVariable(
     model.variables["Electrolyte concentration [mol.m-3]"], cont_sol.t, cont_sol.y, mesh=mesh
)

print('Processed Variable match',
      np.allclose(f_step(step_sol.t[-1], x=x_n, r=r_n),
                  f_cont(cont_sol.t[-1], x=x_n, r=r_n)))

y = amalg[0].y
t = amalg[0].t
for sol in amalg[1:]:
    y = np.column_stack((y, sol.y[:, -1]))
    t = np.concatenate((t, sol.t[-1:]))

print('Amalgamated Solution match', np.allclose(y, cont_sol.y))
