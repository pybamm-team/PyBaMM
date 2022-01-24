#
# Example showing how to load and solve the DFN
#

import pybamm
import numpy as np

models = [
    pybamm.lithium_ion.DFN()  # {"operating mode": "explicit power"}),
    # pybamm.lithium_ion.DFN({"operating mode": "power"}),
    # pybamm.lithium_ion.DFN({"operating mode": "differential power"}),
]

# set parameters and discretise models
for i, model in enumerate(models):
    # create geometry
    params = model.default_parameter_values
    # params.update({"Power function [W]": 4}, check_already_exists=False)
    params.update({"Current function [A]": 0})
    geometry = model.default_geometry
    params.process_model(model)
    params.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
# pybamm.set_logging_level("DEBUG")
solutions = [None] * len(models)
t_eval = np.linspace(0, 3600, 100)
for i, model in enumerate(models):
    print(model.name)
    # solver = pybamm.ScipySolver()
    solver = pybamm.CasadiSolver()
    solutions[i] = solver.solve(model, t_eval)
    # print(solutions[i].solve_time)
    # print(solutions[i].integration_time)
pybamm.dynamic_plot(
    solutions,
    [
        "Terminal voltage [V]",
        "Current [A]",
        "Terminal power [W]",
    ],
)
