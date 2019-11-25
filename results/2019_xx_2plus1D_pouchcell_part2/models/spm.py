import pybamm
import numpy as np


def solve_spm(C_rate=1, t_eval=None, var_pts=None, thermal=False):
    """
    Solves the SPMe and returns variables for plotting.
    """
    options = {}

    if thermal is True:
        options.update({"thermal": "x-lumped"})

    spm = pybamm.lithium_ion.SPM()

    param = spm.default_parameter_values
    param.update({"C-rate": C_rate})

    # discharge timescale
    if t_eval is None:
        tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
        t_end = 900 / tau
        t_eval = np.linspace(0, t_end, 120)

    sim = pybamm.Simulation(spm, parameter_values=param, var_pts=var_pts)
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

    av_cc_current = pybamm.ProcessedVariable(
        sim.built_model.variables["Current collector current density [A.m-2]"], t, y
    )(t)

    T_vol_av = pybamm.ProcessedVariable(
        sim.built_model.variables["Volume-averaged cell temperature [K]"], t, y
    )

    plotting_variables = {
        "Terminal voltage [V]": terminal_voltage,
        "Time [h]": time,
        "Discharge capacity [A.h]": discharge_capacity,
        "Average local current density [A.m-2]": av_cc_current,
        "Volume-averaged cell temperature [K]": T_vol_av(t),
    }

    return plotting_variables

