#
# Compare lead-acid battery models
#
import argparse
import numpy as np
import pybamm
import sys

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
    pybamm.lead_acid.LOQS(
        {"surface form": "differential", "bc_options": {"dimensionality": 1}},
        name="3D LOQS model",
    ),
    pybamm.lead_acid.LOQS(),
    # pybamm.lead_acid.FOQS(),
    # pybamm.lead_acid.CompositeExtended(),
    # # pybamm.lead_acid.Composite({"surface form": "algebraic"}),
    # pybamm.lead_acid.NewmanTiedemann(),
]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param.update(
    {
        # "Bruggeman coefficient": 0.001,
        "Typical current [A]": 20,
        "Initial State of Charge": 1,
        "Typical electrolyte concentration [mol.m-3]": 5600,
        "Negative electrode reference exchange-current density [A.m-2]": 0.08,
        "Positive electrode reference exchange-current density [A.m-2]": 0.006,
    }
)
for model in models:
    param.process_model(model)

# set mesh

# discretise models
sys.setrecursionlimit(10000)
for model in models:
    geometry = model.default_geometry
    param.process_geometry(geometry)
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 25, var.x_s: 41, var.x_p: 34, var.y: 10, var.z: 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 1, 100)
for i, model in enumerate(models):
    solution = model.default_solver.solve(model, t_eval)
    solutions[i] = solution

# plot
output_variables = [
    # [
    #     "Average negative electrode interfacial current density [A.m-2]",
    #     "Average positive electrode interfacial current density [A.m-2]",
    # ],
    # "Average negative electrode surface potential difference [V]",
    # "Average positive electrode surface potential difference [V]",
    # "Electrolyte concentration",
    # "Electrolyte flux",
    "Terminal voltage [V]"
]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)
plot.dynamic_plot()
