#
# Simulations
#
import pickle
import pybamm


def model_comparison(models, Crates, t_eval, extra_parameter_values=None):
    " Solve models at a range of Crates "
    # load parameter values and geometry
    geometry = models[0].default_geometry
    extra_parameter_values = extra_parameter_values or {}
    param = models[0].default_parameter_values
    param.update(extra_parameter_values)

    # Process parameters (same parameters for all models)
    for model in models:
        param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20}
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)

    # discretise models
    discs = {}
    for model in models:
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        # Store discretisation
        discs[model] = disc

    # solve model for range of Crates
    all_variables = {}
    for Crate in Crates:
        all_variables[Crate] = {}
        current = Crate * 17
        pybamm.logger.info("Setting typical current to {} A".format(current))
        param.update({"Typical current [A]": current})
        for model in models:
            param.update_model(model, discs[model])
            solution = model.default_solver.solve(model, t_eval)
            vars = pybamm.post_process_variables(
                model.variables, solution.t, solution.y, mesh
            )
            vars["solution"] = solution
            all_variables[Crate][model.name] = vars

    return all_variables, t_eval


def convergence_study(models, Crate, t_eval, all_npts, save_folder=None):
    " Solve models at a range of number of grid points "
    # load parameter values and geometry
    geometry = models[0].default_geometry
    param = models[0].default_parameter_values
    current = Crate * 17
    # Update parameters with a different porosity
    param.update(
        {
            "Typical current [A]": current,
            "Maximum porosity of negative electrode": 0.92,
            "Maximum porosity of separator": 0.92,
            "Maximum porosity of positive electrode": 0.92,
        }
    )

    # Process parameters (same parameters for all models)
    for model in models:
        param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var = pybamm.standard_spatial_vars

    # solve model for range of Crates
    for npts in all_npts:
        model_variables = {}
        pybamm.logger.info("Setting number of grid points to {}".format(npts))
        var_pts = {var.x_n: npts, var.x_s: npts, var.x_p: npts}
        mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)

        # discretise models
        for model in models:
            model_disc = disc.process_model(model, inplace=False)
            solution = model.default_solver.solve(model_disc, t_eval)
            variables = pybamm.post_process_variables(
                model_disc.variables, solution.t, solution.y, mesh
            )
            variables["solution"] = solution
            model_variables[model.name] = variables

        filename = save_folder + "Crate={}_npts={}.pickle".format(Crate, npts)
        with open(filename, "wb") as f:
            pickle.dump(model_variables, f, pickle.HIGHEST_PROTOCOL)


def simulation(models, t_eval, extra_parameter_values=None, disc_only=False):

    # create geometry
    geometry = models[-1].default_geometry

    # load parameter values and process models and geometry
    param = models[0].default_parameter_values
    extra_parameter_values = extra_parameter_values or {}
    param.update(extra_parameter_values)
    for model in models:
        param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 25, var.x_s: 41, var.x_p: 34}
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)

    # discretise models
    for model in models:
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

    if disc_only:
        return model, mesh

    # solve model
    solutions = [None] * len(models)
    for i, model in enumerate(models):
        solution = model.default_solver.solve(model, t_eval)
        solutions[i] = solution

    return models, mesh, solutions
