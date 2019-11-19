import pybamm
import numpy as np


def solve_spmecc(C_rate=1, t_eval=None, var_pts=None, thermal=False):
    """
    Solves the SPMeCC and returns variables for plotting.
    """

    options = {}

    if thermal is True:
        options.update({"thermal": "x-lumped"})

    # solve the 1D spme
    spme = pybamm.lithium_ion.SPMe(options)


    param = spme.default_parameter_values
    param.update({"C-rate": C_rate})
    # make current collectors not so conductive, just for illustrative purposes
    param.update(
        {
            "Negative current collector conductivity [S.m-1]": 5.96e6,
            "Positive current collector conductivity [S.m-1]": 3.55e6,
        }
    )

    # discharge timescale
    if t_eval is None:
        tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
        t_end = 900 / tau
        t_eval = np.linspace(0, t_end, 120)

    sim_spme = pybamm.Simulation(spme, parameter_values=param, var_pts=var_pts)
    sim_spme.solve(t_eval=t_eval)

    # solve for the current collector
    cc, cc_solution = solve_cc(var_pts, param)

    # get variables for plotting
    t = sim_spme.solution.t
    y_spme = sim_spme.solution.y
    y_cc = cc_solution.y

    time = pybamm.ProcessedVariable(
        sim_spme.built_model.variables["Time [h]"], t, y_spme
    )(t)
    discharge_capacity = pybamm.ProcessedVariable(
        sim_spme.built_model.variables["Discharge capacity [A.h]"], t, y_spme
    )(t)
    current = pybamm.ProcessedVariable(
        sim_spme.built_model.variables["Current [A]"], t, y_spme
    )(t)

    R_cc = param.process_symbol(
        cc.variables["Effective current collector resistance [Ohm]"]
    ).evaluate(t=cc_solution.t, y=y_cc)[0][0]
    delta = param.evaluate(pybamm.standard_parameters_lithium_ion.delta)
    cc_ohmic_losses = -delta * current * R_cc

    terminal_voltage = (
        pybamm.ProcessedVariable(
            sim_spme.built_model.variables["Terminal voltage [V]"], t, y_spme
        )(t)
        + cc_ohmic_losses
    )

    plotting_variables = {
        "Terminal voltage [V]": terminal_voltage,
        "Time [h]": time,
        "Discharge capacity [A.h]": discharge_capacity,
        "Average current collector ohmic losses [Ohm]": cc_ohmic_losses,
    }

    return plotting_variables


def solve_cc(var_pts, param):
    """
    Solving in a separate function as EffectiveResistance2D does not conform
    to the submodel structure.
    """

    model = pybamm.current_collector.EffectiveResistance2D()

    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    solution = model.default_solver.solve(model)

    return model, solution
