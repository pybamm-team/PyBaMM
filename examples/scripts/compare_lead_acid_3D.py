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
    # pybamm.lead_acid.NewmanTiedemann(
    #     {
    #         "surface form": "algebraic",
    #         "current collector": "potential pair",
    #         "dimensionality": 1,
    #     },
    #     name="1+1D NewmanTiedemann",
    # ),
    # pybamm.lead_acid.NewmanTiedemann(
    #     {"surface form": "algebraic", "dimensionality": 1}, name="1D NewmanTiedemann"
    # ),
    # pybamm.lead_acid.CompositeExtended(
    #     {
    #         # "surface form": "algebraic",
    #         "current collector": "potential pair quite conductive",
    #         "dimensionality": 1,
    #     },
    #     name="1+1D composite",
    # ),
    pybamm.lead_acid.CompositeExtended(
        {
            # "surface form": "algebraic",
            "current collector": "potential pair quite conductive averaged",
            "dimensionality": 1,
        },
        name="1+1D composite averaged",
    ),
    # pybamm.lead_acid.CompositeExtended(
    #     {
    #         # "surface form": "algebraic",
    #         # "current collector": "potential pair quite conductive averaged",
    #         "dimensionality": 1
    #     },
    #     name="1D composite",
    # ),
    # pybamm.lead_acid.CompositeExtended(
    #     {
    #         "current collector": "potential pair quite conductive averaged",
    #         "dimensionality": 1,
    #     },
    #     name="1+1D composite quite conductive averaged",
    # ),
    # pybamm.lead_acid.CompositeExtended(
    #     {
    #         # "current collector": "potential pair quite conductive averaged",
    #         "dimensionality": 1
    #     },
    #     name="1D composite",
    # ),
    # pybamm.lead_acid.FOQS(
    #     {"current collector": "potential pair", "dimensionality": 1}, name="1+1D FOQS"
    # ),
    # # pybamm.lead_acid.Composite({"dimensionality": 1}, name="composite"),
    # pybamm.lead_acid.LOQS(
    #     {
    #         # "surface form": "algebraic",
    #         "current collector": "potential pair",
    #         "dimensionality": 1,
    #     },
    #     name="1+1D LOQS",
    # ),
    # pybamm.lead_acid.LOQS({"dimensionality": 1}, name="LOQS"),
]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param.update(
    {
        "Typical current [A]": 20,
        "Bruggeman  coefficient": 0.001,
        "Initial State of Charge": 1,
        "Positive electrode conductivity [S.m-1]": 10 * 8000,
    }
)
for model in models:
    param.process_model(model)

# discretise models
for model in models:
    geometry = model.default_geometry
    param.process_geometry(geometry)
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.y: 10, var.z: 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 3, 1000)
for i, model in enumerate(models):
    solution = model.default_solver.solve(model, t_eval)
    solutions[i] = solution

# plot
V0 = pybamm.ProcessedVariable(
    model.variables["Average open circuit voltage [V]"]
    + model.variables["Average concentration overpotential [V]"]
    + model.variables["Average electrolyte ohmic losses [V]"]
    + model.variables["Average reaction overpotential [V]"]
    + model.variables["Current collector overpotential [V]"],
    solutions[0].t,
    solutions[0].y,
    mesh,
)
V1 = pybamm.ProcessedVariable(
    model.variables["Terminal voltage [V]"], solutions[0].t, solutions[0].y, mesh
)
import ipdb

ipdb.set_trace()


output_variables = [
    # "Local current collector potential difference [V]",
    # "Negative current collector potential [V]",
    # "Positive current collector potential [V]",
    # "X-averaged electrolyte concentration",
    # # "Leading-order current collector current density",
    "Average open circuit voltage [V]",
    "Average concentration overpotential [V]",
    "Average electrolyte ohmic losses [V]",
    "Average reaction overpotential [V]",
    "Current collector overpotential [V]",
    "Current collector current density",
    "Terminal voltage [V]",
]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)
plot.dynamic_plot()
