import pybamm
import numpy as np


def solve_reduced_2p1(C_rate=1, t_eval=None, thermal=False, var_pts=None):

    options = {
        "current collector": "potential pair",
        "dimensionality": 2,
    }

    if thermal is True:
        options.update({"thermal": "x-lumped"})

    model = pybamm.lithium_ion.SPMe(options=options)

    param = model.default_parameter_values
    param.update({"C-rate": C_rate, "Heat transfer coefficient [W.m-2.K-1]": 0.1})

    # discharge timescale
    if t_eval is None:
        tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
        t_end = 900 / tau
        t_eval = np.linspace(0, t_end, 120)

    solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6)
    # solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6)

    sim = pybamm.Simulation(
        model, parameter_values=param, var_pts=var_pts, solver=solver
    )

    sim.solve(t_eval=t_eval)

    t = sim.solution.t
    y = sim.solution.y

    # get variables for plotting
    t = sim.solution.t
    y = sim.solution.y

    time = pybamm.ProcessedVariable(sim.built_model.variables["Time [h]"], t, y)(t)
    discharge_capacity = pybamm.ProcessedVariable(
        sim.built_model.variables["Discharge capacity [A.h]"], t, y
    )(t)
    terminal_voltage = pybamm.ProcessedVariable(
        sim.built_model.variables["Terminal voltage [V]"], t, y
    )(t)

    plotting_variables = {
        "Terminal voltage [V]": terminal_voltage,
        "Time [h]": time,
        "Discharge capacity [A.h]": discharge_capacity,
    }

    return plotting_variables

