#
# Compare lead-acid battery models
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
    # pybamm.lead_acid.LOQS(
    #     {"current collector": "potential pair", "dimensionality": 2}, name="2+1D LOQS"
    # ),
    pybamm.lead_acid.NewmanTiedemann(
        {"current collector": "potential pair", "dimensionality": 1},
        name="1+1D NewmanTiedemann",
    ),
    # pybamm.lead_acid.NewmanTiedemann(
    #     {"dimensionality": 1}, name="1+1D uniform NewmanTiedemann"
    # ),
    pybamm.lead_acid.Composite(
        {"current collector": "potential pair quite conductive", "dimensionality": 1},
        name="1+1D composite",
    ),
    pybamm.lead_acid.Composite(
        {"current collector": "potential pair", "dimensionality": 1},
        name="1+1D composite",
    ),
    # # pybamm.lead_acid.Composite({"dimensionality": 1}, name="composite"),
    pybamm.lead_acid.LOQS(
        {"current collector": "potential pair", "dimensionality": 1}, name="1+1D LOQS"
    ),
    # pybamm.lead_acid.LOQS({"dimensionality": 1}, name="LOQS"),
]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param.update(
    {
        "Typical current [A]": 1,
        "Bruggeman  coefficient": 0.001,
        "Initial State of Charge": 1,
        "Typical electrolyte concentration [mol.m-3]": 5600,
        "Negative electrode reference exchange-current density [A.m-2]": 0.08,
        "Positive electrode reference exchange-current density [A.m-2]": 0.006,
        "Negative electrode conductivity [S.m-1]": 500000,
        "Positive electrode conductivity [S.m-1]": 500000,
    }
)
for model in models:
    param.process_model(model)

# discretise models
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
t_eval = np.linspace(0, 3, 1000)
for i, model in enumerate(models):
    solution = model.default_solver.solve(model, t_eval)
    solutions[i] = solution
    pybamm.post_process_variables(model.variables, solution.t, solution.y, mesh=mesh)

# plot
output_variables = [
    "Local current collector potential difference [V]",
    "Negative current collector potential [V]",
    "Positive current collector potential [V]",
    "X-averaged electrolyte concentration",
    # "Leading-order current collector current density",
    "Current collector current density",
    "Terminal voltage [V]",
]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)
plot.dynamic_plot()
