import pybamm
import numpy as np


class DFNCC:
    def __init__(self, thermal=False, param=None):
        options = {}
        if thermal is True:
            options.update({"thermal": "x-lumped"})

        self.model = pybamm.lithium_ion.DFN(options)
        self.param = self.model.default_parameter_values

        if param:
            self.param.update(param)

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

        # also solve the R_cc problem
        self.solve_cc(var_pts, self.param)

    def solve_cc(self, var_pts, param):
        """
        Solving in a separate function as EffectiveResistance2D does not conform
        to the submodel structure.
        """

        model = pybamm.BaseModel()
        model.submodels = {
            "current collector": pybamm.current_collector.AverageCurrent(
                pybamm.standard_parameters_lithium_ion
            )
        }
        for sm in model.submodels.values():
            model.variables.update(sm.get_fundamental_variables())
            # don't set coupled variables as that doesn't work yet
            sm.set_algebraic(model.variables)
            sm.set_boundary_conditions(model.variables)
            sm.set_initial_conditions(model.variables)
            model.update(sm)

        param.update({"Typical timescale [s]": 3600})
        param.process_model(model)
        geometry = sm.default_geometry
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, sm.default_submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, sm.default_spatial_methods)
        disc.process_model(model)

        solution = sm.default_solver.solve(model)

        self.cc_model = model
        self.cc_solution = solution
        self.cc_mesh = mesh
        self.cc_param = param

        self.y_cc = solution.y

    def processed_variables(self, variables):
        built_vars = {var: self.sim.built_model.variables[var] for var in variables}
        processed_vars = pybamm.post_process_variables(
            built_vars, self.t, self.y, mesh=self.sim.mesh
        )

        # variables from current collector model
        R_cc = self.param.process_symbol(
            self.cc_model.variables["Effective current collector resistance [Ohm]"]
        ).evaluate(t=0.0, y=self.y_cc)[0][0]
        current = pybamm.ProcessedVariable(
            self.sim.built_model.variables["Current [A]"], self.t, self.y
        )

        for var in variables:
            if var == "Terminal voltage [V]":
                internal_V = processed_vars["Terminal voltage [V]"]

                def terminal_voltage(t):
                    cc_ohmic_losses = -current(t) * R_cc
                    return internal_V(t) + cc_ohmic_losses

                processed_vars["Terminal voltage [V]"] = terminal_voltage

        return processed_vars


def solve_dfncc(C_rate=1, t_eval=None, var_pts=None, thermal=False, params=None):
    """
    Solves the SPMeCC and returns variables for plotting.
    """

    options = {}

    if thermal is True:
        options.update({"thermal": "x-lumped"})

    # solve the 1D DFN
    spme = pybamm.lithium_ion.DFN(options)

    param = spme.default_parameter_values
    if params:
        param.update(param)
    param.update({"C-rate": C_rate})

    # discharge timescale
    if t_eval is None:
        tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
        t_end = 900 / tau
        t_eval = np.linspace(0, t_end, 120)

    sim_spme = pybamm.Simulation(spme, parameter_values=param, var_pts=var_pts)
    sim_spme.solve(t_eval=t_eval)

    # solve for the current collector
    cc, cc_solution, cc_mesh, cc_param = solve_cc(var_pts, param)

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

    V_av = pybamm.ProcessedVariable(
        sim_spme.built_model.variables["Terminal voltage"], t, y_spme
    )
    I_av = pybamm.ProcessedVariable(
        sim_spme.built_model.variables["Current [A]"], t, y_spme
    )

    R_cc = param.process_symbol(
        cc.variables["Effective current collector resistance [Ohm]"]
    ).evaluate(t=0.0, y=y_cc)[0][0]
    delta = param.evaluate(pybamm.standard_parameters_lithium_ion.delta)
    cc_ohmic_losses = -delta * current * R_cc

    V_av = pybamm.ProcessedVariable(
        sim_spme.built_model.variables["Terminal voltage [V]"], t, y_spme
    )

    av_cc_current = pybamm.ProcessedVariable(
        sim_spme.built_model.variables["Current collector current density [A.m-2]"],
        t,
        y_spme,
    )(t)

    terminal_voltage = V_av(t) + cc_ohmic_losses

    phi_s_n = pybamm.ProcessedVariable(
        cc.variables["Negative current collector potential [V]"],
        cc_solution.t,
        cc_solution.y,
        mesh=cc_mesh,
    )

    phi_s_p_red = pybamm.ProcessedVariable(
        cc.variables["Reduced positive current collector potential [V]"],
        cc_solution.t,
        cc_solution.y,
        mesh=cc_mesh,
    )

    # R_cn = pybamm.ProcessedVariable(
    #     cc.variables["Negative current collector resistance"],
    #     cc_solution.t,
    #     cc_solution.y,
    #     mesh=cc_mesh,
    # )

    # R_cp = pybamm.ProcessedVariable(
    #     cc.variables["Positive current collector resistance"],
    #     cc_solution.t,
    #     cc_solution.y,
    #     mesh=cc_mesh,
    # )

    def phi_s_n_out(t, y, z):
        return phi_s_n(y=y, z=z)

    def phi_s_p(t, y, z):
        return phi_s_p_red(y=y, z=z) + V_av(t) - delta * R_cc * I_av(t)

    def phi_s_p_red_fun(t, y, z):
        return phi_s_p_red(y=y, z=z)

    def V_cc(t, y, z):
        return phi_s_p(t, y, z) - phi_s_n(y=y, z=z)

    c_e = pybamm.ProcessedVariable(
        sim_spme.built_model.variables["Electrolyte concentration"],
        t,
        y_spme,
        mesh=sim_spme.mesh,
    )

    c_e_dim = pybamm.ProcessedVariable(
        sim_spme.built_model.variables["Electrolyte concentration [mol.m-3]"],
        t,
        y_spme,
        mesh=sim_spme.mesh,
    )

    plotting_variables = {
        "Terminal voltage [V]": terminal_voltage,
        "Time [h]": time,
        "Discharge capacity [A.h]": discharge_capacity,
        "Average current collector ohmic losses [Ohm]": cc_ohmic_losses,
        "L_z": param.process_symbol(pybamm.geometric_parameters.L_z).evaluate(),
        "Negative current collector potential [V]": phi_s_n_out,
        "Positive current collector potential [V]": phi_s_p,
        "Reduced positive current collector potential [V]": phi_s_p_red_fun,
        "Local voltage [V]": V_cc,
        "Average local current density [A.m-2]": av_cc_current,
        "YZ-averaged electrolyte concentration": c_e,
        "YZ-averaged electrolyte concentration [mol.m-3]": c_e_dim,
    }

    return plotting_variables

