import pybamm
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")
pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.SPMe()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values

param.update(
    {
        "Negative electrode surface area density [m-1]": 150000.0,
        "Positive electrode surface area density [m-1]": 150000.0,
    }
)


max_neg = param["Maximum concentration in negative electrode [mol.m-3]"]
max_pos = param["Maximum concentration in positive electrode [mol.m-3]"]
a_neg = param["Negative electrode surface area density [m-1]"]
a_pos = param["Positive electrode surface area density [m-1]"]
l_neg = param["Negative electrode thickness [m]"]
l_pos = param["Positive electrode thickness [m]"]
por_neg = param["Negative electrode porosity"]
por_pos = param["Positive electrode porosity"]

param.process_model(model)
param.process_geometry(geometry)

s_var = pybamm.standard_spatial_vars
var_pts = {s_var.x_n: 5, s_var.x_s: 5, s_var.x_p: 5, s_var.r_n: 5, s_var.r_p: 10}
# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.2, 100)
sol = model.default_solver.solve(model, t_eval)

# plot
out_vars = [
    "Negative active lithium",
    "Positive active lithium",
    "X-averaged electrolyte concentration [mol.m-3]",
    "Negative active volume density",
    "Positive active volume density",
    "RX-averaged positive particle concentration [mol.m-3]",
    "RX-averaged negative particle concentration [mol.m-3]",
]
plot = pybamm.QuickPlot(model, mesh, sol, output_variables=out_vars)
plot.dynamic_plot()
keys = list(model.variables.keys())
keys.sort()

xppc = pybamm.ProcessedVariable(model.variables[out_vars[0]], sol.t, sol.y, mesh=mesh)
xnpc = pybamm.ProcessedVariable(model.variables[out_vars[1]], sol.t, sol.y, mesh=mesh)
xec = pybamm.ProcessedVariable(model.variables[out_vars[2]], sol.t, sol.y, mesh=mesh)
rp = np.linspace(0, 1.0, 11)

plt.figure()
rplt = 0.0
#plt.plot(np.ones(len(sol.t)) * max_neg * por_neg, "r--", label="Max Neg")
plt.plot(xnpc(sol.t, r=rplt) * por_neg, "r", label="Neg Li")
#plt.plot(np.ones(len(sol.t)) * max_pos * por_pos, "b--", label="Max Pos")
plt.plot(xppc(sol.t, r=rplt) * por_pos, "b", label="Pos Li")
plt.plot(xec(sol.t, r=rplt), "g", label="Elec Li")
tot = xnpc(sol.t, r=rplt) * por_neg + xppc(sol.t, r=rplt) * por_pos + xec(sol.t, r=rplt)
plt.plot(tot, "k-", label="Total Li")
plt.legend()
