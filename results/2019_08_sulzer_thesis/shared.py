#
# Simulations: discharge of a lead-acid battery
#
import pickle
import pybamm


def options_to_tuple(options):
    bc_options = tuple(options["bc_options"].items())
    side_reactions = tuple(options["side reactions"])
    other_options = tuple(
        {
            k: v
            for k, v in options.items()
            if k not in ["bc_options", "side reactions"]
        }.items()
    )
    return (*bc_options, *side_reactions, *other_options)


def model_comparison(models, Crates, t_eval):
    " Solve models at a range of Crates "
    # load parameter values and geometry
    geometry = models[0].default_geometry
    param = models[0].default_parameter_values

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
            all_variables[Crate][(model.name, options_to_tuple(model.options))] = vars

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
            model_variables[(model.name, options_to_tuple(model.options))] = variables

        filename = save_folder + "Crate={}_npts={}.pickle".format(Crate, npts)
        with open(filename, "wb") as f:
            pickle.dump(model_variables, f, pickle.HIGHEST_PROTOCOL)
