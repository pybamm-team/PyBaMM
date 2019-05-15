#
# Simulations: discharge of a lead-acid battery
#
import matplotlib.pyplot as plt
import numpy as np
import pybamm


def asymptotics_comparison(models, Crates):
    geometry = models[-1].default_geometry

    # load parameter values and process models and geometry
    param = models[0].default_parameter_values
    for model in models:
        param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5}
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)

    # discretise models
    for model in models:
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

    # solve model
    variables = {}
    t_eval = np.linspace(0, 1, 100)
    for i, model in enumerate(models):
        solver = model.default_solver
        solver.solve(model, t_eval)
        variables[model] = pybamm.post_process_variables(
            model.variables, solver.t, solver.y, mesh
        )

    import ipdb

    ipdb.set_trace()
    # Plot


if __name__ == "__main__":
    models = [
        pybamm.lead_acid.LOQS(),
        pybamm.lead_acid.Composite(),
        pybamm.lead_acid.NewmanTiedemann(),
    ]
    Crates = [0.1]
    asymptotics_comparison(models, Crates)
