#
# Compares the full and lumped thermal models for a single layer Li-ion cell
#

import pybamm
import numpy as np

# load model
pybamm.set_logging_level("INFO")

# options = {"thermal": "x-full"}
# full_thermal_model = pybamm.lithium_ion.SPMe(options)
#
# options = {"thermal": "lumped"}
# lumped_thermal_model = pybamm.lithium_ion.SPMe(options)
#
# models = [full_thermal_model, lumped_thermal_model]

models = [
    pybamm.lithium_ion.SPMe({"thermal": "lumped"}, name="1D lumped"),
    pybamm.lithium_ion.SPMe({"thermal": "x-full"}, name="1D full"),
    pybamm.lithium_ion.SPMe(
        {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "lumped",
        },
        name="2+1D lumped",
    ),
    pybamm.lithium_ion.SPMe(
        {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "pouch cell",
        },
        name="2+1D full",
    ),
    pybamm.lithium_ion.SPMe(
        {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "lumped",
        },
        name="1+1D lumped",
    ),
    pybamm.lithium_ion.SPMe(
        {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "pouch cell",
        },
        name="1+1D full",
    ),
]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param.update(
    {
        "C-rate": 1,
        "Heat transfer coefficient [W.m-2.K-1]": 0.1,
        "Negative current collector conductivity [S.m-1]": 1e12,
        "Positive current collector conductivity [S.m-1]": 1e12,
    }
)

for model in models:
    param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 10,
    var.x_s: 10,
    var.x_p: 10,
    var.r_n: 10,
    var.r_p: 10,
    var.y: 5,
    var.z: 5,
}

# discretise models
for model in models:
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 3500, 100)
for i, model in enumerate(models):
    solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, mode="fast")
    solution = solver.solve(model, t_eval)
    solutions[i] = solution

# plot
output_variables = [
    "Terminal voltage [V]",
    "Volume-averaged Ohmic heating [W.m-3]",
    "Volume-averaged irreversible electrochemical heating [W.m-3]",
    "Volume-averaged reversible heating [W.m-3]",
    "Volume-averaged cell temperature [K]",
]
# labels = ["Full thermal model", "Lumped thermal model"]
plot = pybamm.QuickPlot(solutions, output_variables)  # , labels)
plot.dynamic_plot()
