#
# Compare some charging strategies for lithium-ion batteries
#
# 1. CC: Charge at 1A
# 2. CV: Charge at 4.1V
# 3. CP: Charge at 4W
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
    pybamm.lithium_ion.DFN(name="CC DFN"),
    pybamm.lithium_ion.DFN({"operating mode": "voltage"}, name="CV DFN"),
    pybamm.lithium_ion.DFN({"operating mode": "power"}, name="CP DFN"),
]


# load parameter values and process models and geometry
params = [model.default_parameter_values for model in models]

# 1. Charge at 1A
params[0]["Current function [A]"] = -0.1

# # 2. CV: Charge at 4.1V
params[1]["Voltage function [V]"] = 4.1

# 3. CP-CV: Charge at 4W
params[2]["Power function [W]"] = -0.4


solutions = []
t_eval = np.linspace(0, 1, 100)
for model, param in zip(models, params):
    param.process_model(model)
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    solutions.append(model.default_solver.solve(model, t_eval))

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
