import pybamm
import numpy as np


def solve_spm(C_rate=1, t_eval=None):
    """
    Solves the SPMe and returns variables for plotting.
    """

    spm = pybamm.lithium_ion.SPM()

    param = spm.default_parameter_values
    param.update({"C-rate": C_rate})

    # discharge timescale
    if t_eval is None:
        tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
        t_end = 900 / tau
        t_eval = np.linspace(0, t_end, 120)

    sim = pybamm.Simulation(spm, parameter_values=param)
    sim.solve(t_eval=t_eval)

    mesh = sim.mesh
    t = sim.solution.t
    y = sim.solution.y

    processed_variables = pybamm.post_process_variables(
        sim.built_model.variables, t, y, mesh=mesh
    )

    return processed_variables

