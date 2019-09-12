#
# Simulations
#
import numpy as np
import pybamm
import pickle

variables_to_keep = [
    "x",
    "x [m]",
    "z",
    "z [m]",
    "Time",
    "Time [h]",
    "Average battery open circuit voltage [V]",
    "Average battery reaction overpotential [V]",
    "Average battery concentration overpotential [V]",
    "Average battery electrolyte ohmic losses [V]",
    "Battery current collector overpotential [V]",
    "Battery voltage [V]",
    "Electrolyte concentration [Molar]",
    "X-averaged electrolyte concentration [Molar]",
    "Oxygen concentration [Molar]",
    "X-averaged oxygen concentration [Molar]",
    "Electrolyte potential [V]",
    "X-averaged electrolyte potential [V]",
    "Current collector current density",
    "State of Charge",
    "Fractional Charge Input",
]


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
    var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.z: 10}
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
        # Keep only voltage
        model.variables = {
            "Battery voltage [V]": model.variables["Battery voltage [V]"]
        }
        # Remove voltage cut off
        model.events = {
            name: event
            for name, event in model.events.items()
            if name != "Minimum voltage"
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
                        import ipdb

                        ipdb.set_trace()
                        voltage = np.nan * np.ones_like(t_eval)
                else:
                    voltage = np.nan * np.ones_like(t_eval)
                model_voltages[model.name][Crate][sigma] = voltage

    return model_voltages


def time_comparison(
    models, Crate, sigma, all_npts, t_eval, extra_parameter_values=None
):
    " Solve models with different number of grid points and record the time taken"
    model_times = {model.name: {npts: {} for npts in all_npts} for model in models}
    # load parameter values
    param = models[0].default_parameter_values
    # Update parameters
    extra_parameter_values = extra_parameter_values or {}
    param.update(
        {
            "Typical current [A]": Crate * 17,
            "Positive electrode conductivity [S.m-1]": sigma,
            **extra_parameter_values,
        }
    )

    # set mesh
    var = pybamm.standard_spatial_vars

    # discretise models, store discretisation
    geometries = {}
    for model in models:
        # Remove all variables
        model.variables = {}
        # Remove voltage cut off
        model.events = {
            name: event
            for name, event in model.events.items()
            if name != "Minimum voltage"
        }
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)
        geometries[model] = geometry

    for npts in all_npts:
        pybamm.logger.info("Changing npts to {}".format(npts))
        for model in models:
            if npts > 40 and model.name in ["1+1D Full", "1+1D Composite"]:
                time = np.nan
            else:
                # solve model for range of Crates and npts
                var_pts = {var.x_n: npts, var.x_s: npts, var.x_p: npts, var.z: 20}
                mesh = pybamm.Mesh(
                    geometries[model], model.default_submesh_types, var_pts
                )
                disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
                model_disc = disc.process_model(model, inplace=False)

                try:
                    solution = model.default_solver.solve(model_disc, t_eval)
                    time = solution.solve_time
                except pybamm.SolverError:
                    pybamm.logger.error(
                        "Could not solve {!s} at {} A with sigma={}".format(
                            model.name, current, sigma
                        )
                    )
                    time = np.nan
            model_times[model.name][npts] = time

    return model_times
