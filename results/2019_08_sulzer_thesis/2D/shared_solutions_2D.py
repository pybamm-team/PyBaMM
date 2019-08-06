#
# Simulations
#
import numpy as np
import pybamm
import pickle


def model_comparison(
    models,
    Crates,
    sigmas,
    t_eval,
    savefile,
    use_force=False,
    extra_parameter_values=None,
):
    " Solve models at a range of Crates "
    # Load the models that we know
    all_variables = {Crate: {sigma: {} for sigma in sigmas} for Crate in Crates}
    if use_force is False:
        try:
            with open(savefile, "rb") as f:
                (existing_solutions, t_eval) = pickle.load(f)
            if (
                list(existing_solutions.keys()) == Crates
                and list(existing_solutions[Crates[0]].keys()) == sigmas
            ):
                return existing_solutions, t_eval
        except FileNotFoundError:
            existing_solutions = {}
        for Crate in all_variables.keys():
            if Crate in existing_solutions:
                all_variables[Crate].update(existing_solutions[Crate])
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
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.z: 10}
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)

    # discretise models
    discs = {}
    for model in models:
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        # Store discretisation
        discs[model] = disc

    # solve model for range of Crates
    for Crate in Crates:
        current = Crate * 17
        for sigma in sigmas:
            if all_variables[Crate][sigma] == {}:
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

    with open(savefile, "wb") as f:
        data = (all_variables, t_eval)
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return all_variables, t_eval


def error_comparison(models, Crates, sigmas, t_eval, extra_parameter_values=None):
    " Solve models at differen Crates and sigmas and record the voltage "
    model_voltages = {
        model.name: {Crate: {sigma: {} for sigma in sigmas} for Crate in Crates}
        for model in models
    }
    # load parameter values
    param = models[0].default_parameter_values
    # Update parameters
    extra_parameter_values = extra_parameter_values or {}
    param.update(extra_parameter_values)

    # set mesh
    var = pybamm.standard_spatial_vars

    # solve model for range of Crates and npts
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.z: 5}

    # discretise models, store discretisation
    discs = {}
    for model in models:
        model.variables = {
            "Battery voltage [V]": model.variables["Battery voltage [V]"]
        }
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        discs[model] = disc

    for Crate in Crates:
        current = Crate * 17
        for sigma in sigmas:
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
                try:
                    solution = model.default_solver.solve(model, t_eval)
                    success = True
                except pybamm.SolverError:
                    pybamm.logger.error(
                        "Could not solve {!s} at {} A with sigma={}".format(
                            model.name, current, sigma
                        )
                    )
                    solution = "Could not solve {!s} at {} A with sigma={}".format(
                        model.name, current, sigma
                    )
                    success = False
                if success:
                    try:
                        voltage = pybamm.ProcessedVariable(
                            model.variables["Battery voltage [V]"],
                            solution.t,
                            solution.y,
                            mesh,
                        )(t_eval)
                    except ValueError:
                        voltage = np.nan * np.ones_like(t_eval)
                else:
                    voltage = np.nan * np.ones_like(t_eval)
                model_voltages[model.name][Crate][sigma] = voltage

    return model_voltages


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

        current = Crate * 17
        pybamm.logger.info("Setting typical current to {} A".format(current))
        param.update({"Typical current [A]": current})
        for model in models:
            model_disc = models_disc[model.name]
            disc = discs[model.name]
            param.update_model(model_disc, disc)
            try:
                solution = model.default_solver.solve(model_disc, t_eval)
                success = True
            except pybamm.SolverError:
                pybamm.logger.error(
                    "Could not solve {!s} at {} A with {} points".format(
                        model.name, current, npts
                    )
                )
                solution = "Could not solve {!s} at {} A with {} points".format(
                    model.name, current, npts
                )
                success = False
            if success:
                voltage = pybamm.ProcessedVariable(
                    model_disc.variables["Battery voltage [V]"], solution.t, solution.y
                )(t_eval)
            else:
                voltage = None
            variables = {"Battery voltage [V]": voltage, "solution object": solution}
            models_times_and_voltages[model.name][npts] = variables

    return models_times_and_voltages
