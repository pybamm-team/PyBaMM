import numpy as np
import pybamm

pybamm.set_logging_level("DEBUG")

# load models
models = [
    # pybamm.lead_acid.LOQS(
    #     {
    #         "capacitance": "differential",
    #         "side reactions": ["oxygen"],
    #         "interfacial surface area": "varying",
    #     }
    # ),
    pybamm.lead_acid.LOQS(
        {"capacitance": "differential", "side reactions": ["oxygen"]}
    ),
    pybamm.lead_acid.LOQS(
        {"capacitance": "differential"}  # , "interfacial surface area": "varying"}
    ),
]

# create geometry
geometry = models[-1].default_geometry

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param.update(
    {
        "Typical current [A]": -1,
        "Initial State of Charge": 0.5,
        "Typical electrolyte concentration [mol.m-3]": 5600,
        "Negative electrode reference exchange-current density [A.m-2]": 0.08,
        "Positive electrode reference exchange-current density [A.m-2]": 0.006,
    }
)
for model in models:
    param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30}
mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)

# discretise models
for model in models:
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 0.5, 100)
for i, model in enumerate(models):
    solution = model.default_solver.solve(model, t_eval)
    solutions[i] = solution

# plot
output_variables = [
    "Average negative electrode interfacial current density per volume",
    "Average positive electrode interfacial current density per volume",
    "Average negative electrode oxygen interfacial current density per volume",
    "Average positive electrode oxygen interfacial current density per volume",
    "Average electrolyte concentration [mol.m-3]",
    "Average negative electrode surface area density (main reaction)",
    "Average positive electrode surface area density (main reaction)",
    "Average oxygen concentration [mol.m-3]",
    "Average electrolyte potential [V]",
    "Terminal voltage [V]",
    "Porosity",
]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)
plot.dynamic_plot()
