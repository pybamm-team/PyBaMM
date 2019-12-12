import pybamm
import numpy as np
import sys

import scipy.interpolate as interp


class DFN_2p1D:
    def __init__(self, thermal=False):

        sys.setrecursionlimit(10000)
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
        }
        if thermal is True:
            options.update({"thermal": "x-lumped"})

        self.model = pybamm.lithium_ion.DFN(options)
        self.param = self.model.default_parameter_values

    def solve(self, var_pts, C_rate=1, t_eval=None):

        # discharge timescale
        if t_eval is None:
            tau = self.param.evaluate(
                pybamm.standard_parameters_lithium_ion.tau_discharge
            )
            t_end = 900 / tau
            t_eval = np.linspace(0, t_end, 120)

        self.sim = pybamm.Simulation(
            self.model, parameter_values=self.param, var_pts=var_pts, C_rate=C_rate
        )
        self.sim.solve(t_eval=t_eval)

        self.t = self.sim.solution.t
        self.y = self.sim.solution.y

    def processed_variables(self, variables):
        built_vars = {var: self.sim.built_model.variables[var] for var in variables}
        processed_vars = pybamm.post_process_variables(
            built_vars, self.t, self.y, mesh=self.sim.mesh
        )
        return processed_vars


def solve_2p1D_dfn(C_rate=1, t_eval=None, thermal=False, var_pts=None):

    sys.setrecursionlimit(10000)
    options = {
        "current collector": "potential pair",
        "dimensionality": 2,
    }

    if thermal is True:
        options.update({"thermal": "x-lumped"})

    model = pybamm.lithium_ion.DFN(options=options)

    param = model.default_parameter_values
    param.update({"C-rate": C_rate})

    # discharge timescale
    if t_eval is None:
        tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
        t_end = 900 / tau
        t_eval = np.linspace(0, t_end, 120)

    solver = pybamm.IDAKLUSolver(atol=1e-3, rtol=1e-3, root_tol=1e-3)
    # solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6, root_method="krylov")

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

    phi_s_n_dim = pybamm.ProcessedVariable(
        sim.built_model.variables["Negative current collector potential [V]"],
        t,
        y,
        mesh=sim.mesh,
    )
    phi_s_p_dim = pybamm.ProcessedVariable(
        sim.built_model.variables["Positive current collector potential [V]"],
        t,
        y,
        mesh=sim.mesh,
    )
    V_loc = pybamm.ProcessedVariable(
        sim.built_model.variables["Local voltage [V]"], t, y, mesh=sim.mesh
    )

    V = pybamm.ProcessedVariable(
        sim.built_model.variables["Terminal voltage [V]"], t, y
    )

    def phi_s_p_reduced(t, y, z):
        return phi_s_p_dim(t=t, y=y, z=z) - V(t)

    I_density = pybamm.ProcessedVariable(
        sim.built_model.variables["Current collector current density [A.m-2]"],
        t,
        y,
        mesh=sim.mesh,
    )

    def av_cc_density(t):
        I = I_density(t=t, y=np.linspace(0, 1.5, 100), z=np.linspace(0, 1, 100))
        I_av = np.mean(np.mean(I, axis=0), axis=0)
        return I_av

    T_av = pybamm.ProcessedVariable(
        sim.built_model.variables["X-averaged cell temperature [K]"],
        t,
        y,
        mesh=sim.mesh,
    )

    T_vol_av = pybamm.ProcessedVariable(
        sim.built_model.variables["Volume-averaged cell temperature [K]"],
        t,
        y,
        mesh=sim.mesh,
    )

    c_s_n_surf_av = pybamm.ProcessedVariable(
        sim.built_model.variables["X-averaged negative particle surface concentration"],
        t,
        y,
        mesh=sim.mesh,
    )
    c_s_p_surf_av = pybamm.ProcessedVariable(
        sim.built_model.variables["X-averaged positive particle surface concentration"],
        t,
        y,
        mesh=sim.mesh,
    )

    # c_s_n_surf_yz_av = pybamm.ProcessedVariable(
    #     sim.built_model.variables[
    #         "YZ-averaged negative particle surface concentration"
    #     ],
    #     t,
    #     y,
    #     mesh=sim.mesh,
    # )

    # c_s_p_surf_yz_av = pybamm.ProcessedVariable(
    #     sim.built_model.variables[
    #         "YZ-averaged positive particle surface concentration"
    #     ],
    #     t,
    #     y,
    #     mesh=sim.mesh,
    # )

    # def c_s_n_surf_yz_av(t, x):
    #     y = np.linspace(0, 1.5, 100)
    #     z = np.linspace(0, 1, 100)
    #     return np.mean(np.mean(c_s_n_surf(t=t, x=x, y=y, z=z), axis=1), axis=1)

    # def c_s_p_surf_yz_av(t, x):
    #     y = np.linspace(0, 1.5, 100)
    #     z = np.linspace(0, 1, 100)
    #     return np.mean(np.mean(c_s_p_surf(t=t, x=x, y=y, z=z), axis=1), axis=1)

    # def c_s_n_surf_vol_av(t):
    #     x = np.linspace(0, 1, 100)
    #     y = np.linspace(0, 1.5, 100)
    #     z = np.linspace(0, 1, 100)
    #     return np.mean(
    #         np.mean(np.mean(c_s_n_surf(t=t, x=x, y=y, z=z), axis=0), axis=0), axis=0
    #     )

    # def c_s_p_surf_vol_av(t):
    #     x = np.linspace(0, 1, 100)
    #     y = np.linspace(0, 1.5, 100)
    #     z = np.linspace(0, 1, 100)
    #     return np.mean(
    #         np.mean(np.mean(c_s_p_surf(t=t, x=x, y=y, z=z), axis=0), axis=0), axis=0
    #     )

    y_pts = var_pts[pybamm.standard_spatial_vars.y]
    z_pts = var_pts[pybamm.standard_spatial_vars.z]

    c_e_yz_av = get_yz_average(
        sim.built_model.variables["Electrolyte concentration"],
        sim.mesh,
        t,
        y,
        "whole cell",
        y_pts,
        z_pts,
    )

    c_e_yz_av_dim = get_yz_average(
        sim.built_model.variables["Electrolyte concentration [mol.m-3]"],
        sim.mesh,
        t,
        y,
        "whole cell",
        y_pts,
        z_pts,
    )

    plotting_variables = {
        "Terminal voltage [V]": terminal_voltage,
        "Time [h]": time,
        "Negative current collector potential [V]": phi_s_n_dim,
        "Positive current collector potential [V]": phi_s_p_dim,
        "Reduced positive current collector potential [V]": phi_s_p_reduced,
        "Discharge capacity [A.h]": discharge_capacity,
        "Local voltage [V]": V_loc,
        "L_z": param.process_symbol(pybamm.geometric_parameters.L_z).evaluate(),
        "Local current density [A.m-2]": I_density,
        "Average local current density [A.m-2]": av_cc_density(t),
        "X-averaged cell temperature [K]": T_av,
        "X-averaged negative particle surface concentration": c_s_n_surf_av,
        "X-averaged positive particle surface concentration": c_s_p_surf_av,
        "YZ-averaged electrolyte concentration": c_e_yz_av,
        "YZ-averaged electrolyte concentration [mol.m-3]": c_e_yz_av_dim,
    }

    if thermal:
        plotting_variables.update({"Volume-averaged cell temperature [K]": T_vol_av(t)})

    return plotting_variables


def get_yz_average(var, mesh, t_nodes, sol, domain, ypts, zpts):

    if domain == "whole cell":
        x_nodes = np.concatenate(
            (
                mesh["negative electrode"][0].nodes,
                mesh["separator"][0].nodes,
                mesh["positive electrode"][0].nodes,
            )
        )
    else:
        x_nodes = mesh[domain][0].nodes

    y_edges = np.linspace(0, 1.5, ypts + 1)
    y_nodes = (y_edges[:-1] + y_edges[1:]) / 2

    z_edges = np.linspace(0, 1, zpts + 1)
    z_nodes = (z_edges[:-1] + z_edges[1:]) / 2

    entries = np.zeros((x_nodes.size, t_nodes.size))

    for i, t in enumerate(t_nodes):
        # full_3D = np.reshape(
        #     var.evaluate(t, sol[:, i]), (x_nodes.size, y_nodes.size, z_nodes.size)
        # )
        # yz_averaged = np.mean(np.mean(full_3D, axis=1), axis=1)
        full_3D = np.reshape(
            var.evaluate(t, sol[:, i]), (y_nodes.size, z_nodes.size, x_nodes.size)
        )
        yz_averaged = np.mean(np.mean(full_3D, axis=0), axis=0)
        entries[:, i] = yz_averaged

    extrap_space_left = np.array([2 * x_nodes[0] - x_nodes[1]])
    extrap_space_right = np.array([2 * x_nodes[-1] - x_nodes[-2]])
    x_nodes = np.concatenate([extrap_space_left, x_nodes, extrap_space_right])

    extrap_entries_left = 2 * entries[0] - entries[1]
    extrap_entries_right = 2 * entries[-1] - entries[-2]

    entries = np.vstack([extrap_entries_left, entries, extrap_entries_right])

    interpolation_function = interp.RegularGridInterpolator(
        (x_nodes, t_nodes), entries, method="linear", fill_value=np.nan,
    )

    def fun(t, x):
        return interpolation_function((x, t))

    return fun
