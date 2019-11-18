import pybamm
import numpy as np


def solve_full_2p1(y_pts, z_pts):

    options = {
        "current collector": "potential pair",
        "dimensionality": 2,
        "thermal": "x-lumped",
    }

    model = pybamm.lithium_ion.DFN(options=options)

    param = model.default_parameter_values
    param.update({"C-rate": 2, "Heat transfer coefficient [W.m-2.K-1]": 0.1})

    var_pts = model.default_var_pts
    var_pts.update(
        {pybamm.standard_spatial_vars.y: y_pts, pybamm.standard_spatial_vars.z: z_pts}
    )

    # discharge timescale
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)

    # solve model
    t_end = 900 / tau
    t_eval = np.linspace(0, t_end, 120)
    solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6)
    # solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6)

    sim = pybamm.Simulation(
        model, parameter_values=param, var_pts=var_pts, solver=solver
    )

    sim.solve(t_eval=t_eval)

    mesh = sim.mesh
    t = sim.solution.t
    y = sim.solution.y

    processed_variables = pybamm.post_process_variables(
        model.variables, t, y, mesh=mesh
    )

    return processed_variables

