#
# Simulations
#
import pybamm


def model_comparison(models, Crates, sigmas, t_eval, extra_parameter_values=None):
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
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.z: 5}
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
        for sigma in sigmas:
            all_variables[Crate][sigma] = {}
            pybamm.logger.info(
                """Setting typical current to {} A
                and positive electrode condutivity to {} S/m""".format(
                    current, sigma
                )
            )
            param.update(
                {
                    "Typical current [A]": current,
                    "Positive electrode conductivity [S.m-1]": sigma,
                }
            )
            for model in models:
                param.update_model(model, discs[model])
                solution = model.default_solver.solve(model, t_eval)
                variables = pybamm.post_process_variables(
                    model.variables, solution.t, solution.y, mesh
                )
                variables["solution"] = solution
                all_variables[Crate][sigma][model.name] = variables

    return all_variables, t_eval


def convergence_study(models, Crates, sigmas, t_eval, extra_parameter_values=None):
    " Solve models at a range of number of grid points "
    # load parameter values and geometry
    geometry = models[0].default_geometry
    param = models[0].default_parameter_values
    # Update parameters
    extra_parameter_values = extra_parameter_values or {}
    param.update(extra_parameter_values)

    # Process parameters (same parameters for all models)
    for model in models:
        param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var = pybamm.standard_spatial_vars

    # solve model for range of Crates and npts
    models_times_and_voltages = {model.name: {} for model in models}
    for npts in all_npts:
        pybamm.logger.info("Setting number of grid points to {}".format(npts))
        var_pts = {var.x_n: npts, var.x_s: npts, var.x_p: npts}
        mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)

        # discretise models, store discretised model and discretisation
        models_disc = {}
        discs = {}
        for model in models:
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            models_times_and_voltages[model.name][npts] = {}
            models_disc[model.name] = disc.process_model(model, inplace=False)
            discs[model.name] = disc

        # Solve for a range of C-rates
        for Crate in Crates:
            current = Crate * 17
            pybamm.logger.info("Setting typical current to {} A".format(current))
            param.update({"Typical current [A]": current})
            for model in models:
                model_disc = models_disc[model.name]
                disc = discs[model.name]
                param.update_model(model_disc, disc)
                try:
                    solution = model.default_solver.solve(model_disc, t_eval)
                except pybamm.SolverError:
                    pybamm.logger.error(
                        "Could not solve {!s} at {} A with {} points".format(
                            model.name, current, npts
                        )
                    )
                    continue
                voltage = pybamm.ProcessedVariable(
                    model_disc.variables["Battery voltage [V]"], solution.t, solution.y
                )(t_eval)
                variables = {
                    "Battery voltage [V]": voltage,
                    "solution object": solution,
                }
                models_times_and_voltages[model.name][npts][Crate] = variables

    return models_times_and_voltages
