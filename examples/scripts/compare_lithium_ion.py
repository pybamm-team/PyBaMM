import pybamm
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

# load models
options = {"thermal": None}
pybamm.set_logging_level("INFO")
models = [
    pybamm.lithium_ion.SPM(options),
    pybamm.lithium_ion.SPMe(options),
    pybamm.lithium_ion.DFN(options),
]


# load parameter values and process models and geometry
param = models[0].default_parameter_values
param["Typical current [A]"] = 1.0
for model in models:
    param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}

# discretise models
for model in models:
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 0.17, 100)
for i, model in enumerate(models):
    solutions[i] = model.default_solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(models, mesh, solutions)
plot.dynamic_plot()
