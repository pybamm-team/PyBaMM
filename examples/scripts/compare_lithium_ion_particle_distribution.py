#
# Compare lithium-ion battery models
#
import argparse
import numpy as np
import pybamm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug", action="store_true", help="Set logging level to 'DEBUG'."
)
args = parser.parse_args()
if args.debug:
    pybamm.set_logging_level("DEBUG")
else:
    pybamm.set_logging_level("INFO")

# load models
options = {"thermal": "isothermal"}
models = [
    pybamm.lithium_ion.DFN(options, name="standard DFN"),
    pybamm.lithium_ion.DFN(options, name="particle DFN"),
]


# load parameter values and process models and geometry
params = [models[0].default_parameter_values, models[1].default_parameter_values]
params[0]["Typical current [A]"] = 1.0
params[0].process_model(models[0])


params[1]["Typical current [A]"] = 1.0


def negative_distribution(x):
    return 1 + x


def positive_distribution(x):
    return 1 + (x - (1 - models[1].param.l_p))


params[1]["Negative particle distribution in x"] = negative_distribution
params[1]["Positive particle distribution in x"] = positive_distribution
params[1].process_model(models[1])

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}

# discretise models
for param, model in zip(params, models):
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 3600, 100)
for i, model in enumerate(models):
    solutions[i] = model.default_solver.solve(model, t_eval)


output_variables = [
    "Negative particle surface concentration",
    "Electrolyte concentration",
    "Positive particle surface concentration",
    "Current [A]",
    "Negative electrode potential [V]",
    "Electrolyte potential [V]",
    "Positive electrode potential [V]",
    "Terminal voltage [V]",
    "Negative particle distribution in x",
    "Positive particle distribution in x",
]

# plot
plot = pybamm.QuickPlot(solutions, output_variables=output_variables)
plot.dynamic_plot()
