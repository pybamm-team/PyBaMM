import pybamm
import time
import numpy as np


# This script provides an example for massively vectorised
# model solves using the JAX BDF solver. First,
# we set up the model and process parameters
model = pybamm.lithium_ion.SPM()
model.convert_to_format = "jax"
model.events = []  # remove events (not supported in jax)
geometry = model.default_geometry
param = pybamm.ParameterValues("Chen2020")
param.update({"Current function [A]": "[input]"})
param.process_geometry(geometry)
param.process_model(model)

# Discretise and setup solver
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
t_eval = np.linspace(0, 3600, 100)
solver = pybamm.JaxSolver(atol=1e-6, rtol=1e-6, method="BDF")

# Set number of vectorised solves
values = np.linspace(0.01, 1.0, 1000)
inputs = [{"Current function [A]": value} for value in values]

# Run solve for all inputs, with a just-in-time compilation
# occurring on the first solve. All sequential solves will
# use the compiled code, for a large performance improvement.
start_time = time.time()
solution = solver.solve(model, t_eval, inputs=inputs)
print(f"Time taken: {time.time() - start_time}")  # 1.2s

# Plot one of the solves
plot = pybamm.QuickPlot(
    solution[5],
    [
        "Negative particle concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "Positive particle concentration [mol.m-3]",
        "Current [A]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        "Voltage [V]",
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
