#
# Compare some charging strategies for lithium-ion batteries
#
# 1. CC-CV: Charge at 1A to 4.2V then 4.2V hold
# 2. CV: Charge at 4.1V
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


class CCCV:
    num_switches = 1

    def __call__(self, variables):
        # switch controls 1A charge vs 4.2V charge
        # charging current is negative
        a = variables["Switch 1"]
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return (a < 0.5) * (I + 1) + (a > 0.5) * (I)


class CCCP:
    num_switches = 1

    def __call__(self, variables):
        # switch controls 4W charge vs 4.2V charge
        # charging current (and hence power) is negative
        a = variables["Switch 1"]
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return (a * (V > 0) <= 0) * (I * V + 4) + (a * (V > 0) > 0) * (V - 4.2)


# load models
models = [
    pybamm.lithium_ion.SPM({"operating mode": CCCV}, name="CCCV SPM"),
    # pybamm.lithium_ion.DFN({"operating mode": "voltage"}, name="CV DFN"),
    # pybamm.lithium_ion.DFN({"operating mode": CCCP}, name="CCCP DFN"),
]


# load parameter values and process models and geometry
params = [model.default_parameter_values for model in models]

# 1. CC-CV: Charge at 1C ( A) to 4.2V then 4.2V hold
params[0]["Switch 1"] = 0
params[0]["Upper voltage cut-off [V]"] = 4.15

# # 2. CV: Charge at 4.1V
# params[1]["Voltage function"] = 4.1

# # 3. CP-CV: Charge at 4W to 4.2V then 4.2V hold
# b = pybamm.Parameter("CCCP switch")


# params[2]["CCCP switch"] = 1  # start with CP
# params[2]["External circuit function"] = cccp

for model, param in zip(models, params):
    param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}

# discretise models
discs = {}
for model in models:
    # create geometry
    model.convert_to_format = "python"
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    discs[model] = disc

# solve models
solver0 = model.default_solver
solution0 = solver0.step(model, 1, npts=1000)
params[0]["Switch 1"] = 1
params[0]["Upper voltage cut-off [V]"] = 4.16
params[0].update_model(models[0], discs[models[0]])
solution0.append(solver0.step(model, 1 - solution0.t[-1], npts=1000))
solutions = [solution0]
# # Step
# solution0 = models[0].default_solver.step(models[0], t_eval)
# # Switch to CV
# params[0].update({"CCCV switch": 0})
# params[0].update_model(models[0], discs[models[0]])
# # Step
# solution0.append(models[0].default_solver.step(models[0], t_eval))

# solution1 = models[1].default_solver.solve(models[1], t_eval)

# solutions = [solution0, solution1]
# plot
output_variables = [
    "Negative particle surface concentration",
    "Electrolyte concentration",
    "Positive particle surface concentration",
    "Current [A]",
    "Negative electrode potential [V]",
    "Electrolyte potential [V]",
    "Switch 1",
    "Terminal voltage [V]",
]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)
plot.dynamic_plot()
