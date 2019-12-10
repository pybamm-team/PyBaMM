#
# Compare thermal and isothermal lithium-ion battery models
#
import numpy as np
import pybamm

# load models
options = {"thermal": "isothermal"}
models = [
    pybamm.lithium_ion.DFN({"thermal": "isothermal"}, name="isothermal"),
    pybamm.lithium_ion.DFN({"thermal": "x-full"}, name="thermal"),
]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
C_rate = 1
param.update({"C-rate": C_rate})
for model in models:
    param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 11, var.x_s: 11, var.x_p: 11, var.r_n: 11, var.r_p: 11}

# discretise models
for model in models:
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
solutions = [None] * len(models)
t_eval = np.linspace(0, 3600 / tau / C_rate, 60)
for i, model in enumerate(models):
    solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6, mode="fast")
    solutions[i] = solver.solve(model, t_eval)

# plot
output_variables = [
    "Negative electrode potential [V]",
    "Positive electrode potential [V]",
    "Negative electrode current density [A.m-2]",
    "Positive electrode current density [A.m-2]",
    "Electrolyte concentration [mol.m-3]",
    "Electrolyte potential [V]",
    "Terminal voltage [V]",
    "Volume-averaged cell temperature [K]",
]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables=output_variables)
plot.dynamic_plot()
