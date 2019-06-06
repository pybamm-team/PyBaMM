#
# Simulations: discharge of a lead-acid battery
#
import pybamm


def options_to_tuple(options):
    bc_options = tuple(options["bc_options"].items())
    other_options = tuple(
        {k: v for k, v in options.items() if k != "bc_options"}.items()
    )
    return (*bc_options, *other_options)


def model_comparison(models, Crates, t_eval):
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
            solver = model.default_solver
            solution = solver.solve(model, t_eval)
            all_variables[Crate][
                (model.name, options_to_tuple(model.options))
            ] = pybamm.post_process_variables(
                model.variables, solution.t, solution.y, mesh
            )

    return all_variables, t_eval
