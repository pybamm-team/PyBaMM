#
# Compare some charging strategies for lithium-ion batteries
#
# 1. CC-CV: Charge at 1A to 4.2V then 4.2V hold
# 2. CV: Charge at 4.2V
# 3. Constant Power-CV: Charge at 4W to 4.2V then 4.2V hold
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
    pybamm.lithium_ion.DFN({"operating mode": "custom"}),
    pybamm.lithium_ion.DFN({"operating mode": "voltage"}),
    pybamm.lithium_ion.DFN({"operating mode": "custom"}),
]


# load parameter values and process models and geometry
params = [models[0].default_parameter_values] * 3

# 1. CC-CV: Charge at 1C ( A) to 4.2V then 4.2V hold
a = pybamm.Parameter("CCCV switch")


def cccv(I, V):
    # switch a controls 1A charge vs 4.2V charge
    # charging current is negative
    return a * (I + 1) + (1 - a) * (V - 4.2)


params[0]["CCCV switch"] = 1  # start with CC
params[0]["External circuit function"] = cccv

# 2. CV: Charge at 4.2V
params[1]["Voltage function"] = 4.2
for model, param in zip(models, params):
    param.process_model(model)

# 3. CP-CV: Charge at 4W to 4.2V then 4.2V hold
b = pybamm.Parameter("CCCP switch")


def cccp(I, V):
    # switch a controls 1A charge vs 4.2V charge
    # charging current is negative
    return b * (I * V + 4) + (1 - b) * (V - 4.2)


params[0]["CCCP switch"] = 1  # start with CP
params[0]["External circuit function"] = cccp

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
t_eval = np.linspace(0, 0.3, 100)
for i, model in enumerate(models):
    solutions[i] = model.default_solver.solve(model, t_eval)

# plot
output_variables = [
    "Negative particle surface concentration",
    "Electrolyte concentration",
    "Positive particle surface concentration",
    "Current [A]",
    "Negative electrode potential [V]",
    "Electrolyte potential [V]",
    "Terminal power [W]",
    "Terminal voltage [V]",
]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)
plot.dynamic_plot()
