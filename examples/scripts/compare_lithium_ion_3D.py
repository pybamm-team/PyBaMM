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
models = [
    pybamm.lithium_ion.SPM(
        {"current collector": "potential pair", "dimensionality": 2}, name="2+1D SPM"
    ),
    pybamm.lithium_ion.SPMe(
        {"current collector": "potential pair", "dimensionality": 2}, name="2+1D SPMe"
    ),
]

# load parameter values and process models
param = models[0].default_parameter_values
for model in models:
    param.process_model(model)

# process geometry and discretise models
for model in models:
    geometry = model.default_geometry
    param.process_geometry(geometry)
    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: 5,
        var.x_s: 5,
        var.x_p: 5,
        var.r_n: 5,
        var.r_p: 5,
        var.y: 5,
        var.z: 5,
    }
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 1, 1000)
for i, model in enumerate(models):
    model.convert_to_format = "casadi"
    solution = model.default_solver.solve(model, t_eval)
    solutions[i] = solution

# plot
# TO DO: plotting 3D variables
output_variables = ["Terminal voltage [V]"]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)
plot.dynamic_plot()
