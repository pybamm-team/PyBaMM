import pybamm
import numpy as np

models = [
    pybamm.lithium_ion.SPM({"particle shape": "spherical"}, name="spherical"),
    pybamm.lithium_ion.SPM({"particle shape": "user"}, name="user"),
]
params = [models[0].default_parameter_values, models[0].default_parameter_values]

# set same mesh for all models
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5}


for model, param in zip(models, params):
    if model.name == "user":
        R_n = param["Negative particle radius [m]"]
        R_p = param["Positive particle radius [m]"]
        eps_s_n = param["Negative electrode active material volume fraction"]
        eps_s_p = param["Positive electrode active material volume fraction"]

        param.update(
            {
                "Negative electrode surface area to volume ratio [m-1]": 3
                * eps_s_n
                / R_n,
                "Positive electrode surface area to volume ratio [m-1]": 3
                * eps_s_p
                / R_p,
                "Negative surface area per unit volume distribution in x": 1,
                "Positive surface area per unit volume distribution in x": 1,
            },
            check_already_exists=False,
        )

    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


# solve model
solutions = []
t_eval = np.linspace(0, 3600, 100)
for model in models:
    solution = pybamm.CasadiSolver().solve(model, t_eval)
    solutions.append(solution)

pybamm.dynamic_plot(solutions)
