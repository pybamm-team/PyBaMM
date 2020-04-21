import pybamm
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt


def make_comsol_model(comsol_variables, mesh, param, z_interp=None, thermal=True):
    "Make Comsol 'model' for comparison"
    print("Start making COMSOL model")

    # comsol time
    comsol_t = comsol_variables["time"]
    interp_kind = "cubic"

    # discharge timescale
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)

    # interpolate using *dimensional* space
    L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)

    if z_interp is None:
        pybamm_z = mesh["current collector"][0].nodes
        z_interp = pybamm_z * L_z

    def get_interp_fun_curr_coll(variable_name):
        """
        Interpolate in space to plotting nodes, and then create function to interpolate
        in time that can be called for plotting at any t.
        """

        comsol_z = comsol_variables[variable_name + "_z"]
        variable = comsol_variables[variable_name]
        try:
            variable = interp.interp1d(comsol_z, variable, axis=0, kind=interp_kind)(
                z_interp
            )
        except:
            import ipdb

            ipdb.set_trace()

        def myinterp(t):
            return interp.interp1d(comsol_t, variable, kind=interp_kind)(t)[
                :, np.newaxis
            ]

        # Make sure to use dimensional time
        fun = pybamm.Function(myinterp, pybamm.t * tau, name=variable_name + "_comsol")
        fun.domain = "current collector"
        return fun

    # Create interpolating functions to put in comsol_model.variables dict
    comsol_voltage = interp.interp1d(
        comsol_t, comsol_variables["voltage"], kind=interp_kind
    )

    comsol_phi_s_cn = get_interp_fun_curr_coll("phi_s_cn")
    comsol_phi_s_cp = get_interp_fun_curr_coll("phi_s_cp")
    comsol_current = get_interp_fun_curr_coll("current")

    # Create comsol model with dictionary of Matrix variables
    comsol_model = pybamm.BaseModel()
    comsol_model.variables = {
        "Terminal voltage [V]": pybamm.Function(
            comsol_voltage, pybamm.t * tau, name="voltage_comsol"
        ),
        "Negative current collector potential [V]": comsol_phi_s_cn,
        "Positive current collector potential [V]": comsol_phi_s_cp,
        "Current collector current density [A.m-2]": comsol_current,
    }

    # Add thermal variables
    if thermal:

        comsol_vol_av_temperature = interp.interp1d(
            comsol_t, comsol_variables["volume-averaged temperature"], kind=interp_kind
        )

        comsol_temperature = get_interp_fun_curr_coll("temperature")
        comsol_model.variables.update(
            {
                "X-averaged cell temperature [K]": comsol_temperature,
                "Volume-averaged cell temperature [K]": pybamm.Function(
                    comsol_vol_av_temperature,
                    pybamm.t * tau,
                    name="av_temperature_comsol",
                ),
            }
        )

    # Add concentrations if provided
    if "c_s_n" in comsol_variables.keys():
        comsol_c_s_n = get_interp_fun_curr_coll("c_s_n")
        comsol_model.variables.update(
            {
                "X-averaged negative particle surface concentration [mol.m-3]": comsol_c_s_n
            }
        )
    if "c_s_p" in comsol_variables.keys():
        comsol_c_s_p = get_interp_fun_curr_coll("c_s_p")
        comsol_model.variables.update(
            {
                "X-averaged positive particle surface concentration [mol.m-3]": comsol_c_s_p
            }
        )

    comsol_model.z_interp = z_interp
    comsol_model.t = comsol_t

    print("Finish making COMSOL model")
    return comsol_model


def plot_t_var(
    var,
    pybamm_model,
    comsol_model,
    mesh,
    solution,
    param,
    plot_times=None,
    plot_error="both",
):

    # Get discharge timescale
    tau = param.process_symbol(
        pybamm.standard_parameters_lithium_ion.tau_discharge
    ).evaluate()

    # Set plot times if not provided
    if plot_times is None:
        plot_times = solution.t * tau

    # Process variables
    pybamm_var = pybamm.ProcessedVariable(
        pybamm_model.variables[var], solution.t, solution.y, mesh=mesh
    )(plot_times / tau)
    comsol_var = pybamm.ProcessedVariable(
        comsol_model.variables[var], solution.t, solution.y, mesh=mesh
    )(plot_times / tau)

    # Make plot

    # add extra row for errors
    if plot_error in ["abs", "rel"]:
        n_rows = 2
    elif plot_error == "both":
        n_rows = 3
    else:
        n_rows = 1
    fig, ax = plt.subplots(n_rows, 1, figsize=(15, 8))

    ax[0].plot(plot_times, pybamm_var, "-", label="PyBaMM")
    ax[0].plot(plot_times, comsol_var, "o", fillstyle="none", label="COMSOL")
    if plot_error == "abs":
        error = np.abs(pybamm_var - comsol_var)
        ax[1].plot(plot_times, error, "-")
    elif plot_error == "rel":
        error = np.abs((pybamm_var - comsol_var) / comsol_var)
        ax[1].plot(plot_times, error, "-")
    elif plot_error == "both":
        abs_error = np.abs(pybamm_var - comsol_var)
        rel_error = np.abs((pybamm_var - comsol_var) / comsol_var)
        ax[1].plot(plot_times, abs_error, "-")
        ax[2].plot(plot_times, rel_error, "-")

    # set labels
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(var)
    if plot_error in ["abs", "rel"]:
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("error (" + plot_error + ")")
    elif plot_error == "both":
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("error (abs)")
        ax[2].set_xlabel("t")
        ax[2].set_ylabel("error (rel)")
    ax[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()


def plot_cc_var(
    var,
    pybamm_model,
    comsol_model,
    mesh,
    solution,
    param,
    plot_times=None,
    plot_error="both",
    scale=None,
):

    # Get discharge timescale
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)

    # Set plot times if not provided
    if plot_times is None:
        plot_times = solution.t * tau

    # Process variables
    z_plot = comsol_model.z_interp  # dimensional
    L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)
    pybamm_var_fun = pybamm.ProcessedVariable(
        pybamm_model.variables[var], solution.t, solution.y, mesh=mesh
    )
    comsol_var_fun = pybamm.ProcessedVariable(
        comsol_model.variables[var], solution.t, solution.y, mesh=mesh
    )
    # If var is positive current collector potential compute relative to
    # voltage
    if var == "Positive current collector potential [V]":
        pybamm_voltage_fun = pybamm.ProcessedVariable(
            pybamm_model.variables["Terminal voltage [V]"],
            solution.t,
            solution.y,
            mesh=mesh,
        )
        comsol_voltage_fun = pybamm.ProcessedVariable(
            comsol_model.variables["Terminal voltage [V]"],
            solution.t,
            solution.y,
            mesh=mesh,
        )

    # add extra cols for errors
    if plot_error in ["abs", "rel"]:
        n_cols = 2
    elif plot_error == "both":
        n_cols = 3
    else:
        n_cols = 1
    fig, ax = plt.subplots(n_cols, 1, figsize=(15, 8))
    cmap = plt.get_cmap("inferno")

    # Loop over plot_times
    for ind, t in enumerate(plot_times):
        color = cmap(float(ind) / len(plot_times))

        pybamm_var = pybamm_var_fun(z=z_plot / L_z, t=t / tau)
        comsol_var = comsol_var_fun(z=z_plot / L_z, t=t / tau)
        # If var is positive current collector potential compute relative to
        # voltage
        if var == "Positive current collector potential [V]":
            pybamm_var = pybamm_var - pybamm_voltage_fun(t=t / tau)
            comsol_var = comsol_var - comsol_voltage_fun(t=t / tau)

        ax[0].plot(
            z_plot,
            comsol_var,
            "o",
            color=color,
            fillstyle="none",
            label="COMSOL" if ind == 0 else "",
        )
        ax[0].plot(
            z_plot, pybamm_var, "-", color=color, label="PyBaMM (t={:.0f} s)".format(t)
        )

        if plot_error == "abs":
            error = np.abs(pybamm_var - comsol_var)
            ax[1].plot(z_plot, error, "-", color=color)
        elif plot_error == "rel":
            if scale is None:
                scale_val = comsol_var
            elif scale == "auto":
                scale_val = np.abs(np.max(comsol_var) - np.min(comsol_var))
            else:
                scale_val = scale
            error = np.abs((pybamm_var - comsol_var) / scale_val)
            ax[1].plot(error, z_plot, "-", color=color)
        elif plot_error == "both":
            abs_error = np.abs(pybamm_var - comsol_var)
            if scale is None:
                scale_val = comsol_var
            elif scale == "auto":
                scale_val = np.abs(np.max(comsol_var) - np.min(comsol_var))
            else:
                scale_val = scale
            rel_error = np.abs((pybamm_var - comsol_var) / scale_val)
            ax[1].plot(z_plot, abs_error, "-", color=color)
            ax[2].plot(z_plot, rel_error, "-", color=color)

    # set labels
    ax[0].set_xlabel("z")
    ax[0].set_ylabel(var)
    if plot_error in ["abs", "rel"]:
        ax[1].set_xlabel("z")
        ax[1].set_ylabel("error (" + plot_error + ")")
    elif plot_error == "both":
        ax[1].set_xlabel("z")
        ax[1].set_ylabel("error (abs)")
        ax[2].set_xlabel("z")
        ax[2].set_ylabel("error (rel)")

    ax[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()


def plot_cc_potentials(
    pybamm_model, comsol_model, mesh, solution, param, plot_times=None, sharex=False
):

    # Get discharge timescale
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)

    # Set plot times if not provided
    if plot_times is None:
        plot_times = solution.t * tau

    # Process variables
    z_plot = comsol_model.z_interp  # dimensional
    L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)

    pybamm_phi_s_cn_fun = pybamm.ProcessedVariable(
        pybamm_model.variables["Negative current collector potential [V]"],
        solution.t,
        solution.y,
        mesh=mesh,
    )
    pybamm_phi_s_cp_fun = pybamm.ProcessedVariable(
        pybamm_model.variables["Positive current collector potential [V]"],
        solution.t,
        solution.y,
        mesh=mesh,
    )
    comsol_phi_s_cn_fun = pybamm.ProcessedVariable(
        comsol_model.variables["Negative current collector potential [V]"],
        solution.t,
        solution.y,
        mesh=mesh,
    )
    comsol_phi_s_cp_fun = pybamm.ProcessedVariable(
        comsol_model.variables["Positive current collector potential [V]"],
        solution.t,
        solution.y,
        mesh=mesh,
    )

    # Make plot
    fig, ax = plt.subplots(2, 2, sharex=sharex, figsize=(6.4, 4))
    fig.subplots_adjust(
        left=0.1, bottom=0.1, right=0.95, top=0.85, wspace=0.3, hspace=0.5
    )
    cmap = plt.get_cmap("inferno")

    # Loop over plot_times
    for ind, t in enumerate(plot_times):
        color = cmap(float(ind) / len(plot_times))

        pybamm_phi_s_cn = pybamm_phi_s_cn_fun(z=z_plot / L_z, t=t / tau)
        pybamm_phi_s_cp = pybamm_phi_s_cp_fun(z=z_plot / L_z, t=t / tau)
        comsol_phi_s_cn = comsol_phi_s_cn_fun(z=z_plot / L_z, t=t / tau)
        comsol_phi_s_cp = comsol_phi_s_cp_fun(z=z_plot / L_z, t=t / tau)

        ax[0, 0].plot(
            z_plot[0::9] * 1e3,
            comsol_phi_s_cn[0::9],
            "o",
            color=color,
            fillstyle="none",
            label="COMSOL" if ind == 0 else "",
        )
        ax[0, 0].plot(
            z_plot * 1e3,
            pybamm_phi_s_cn,
            "-",
            color=color,
            label="PyBaMM" if ind == 0 else "",
        )
        error = np.abs(pybamm_phi_s_cn - comsol_phi_s_cn)
        ax[1, 0].plot(z_plot * 1e3, error, "-", color=color)
        ax[0, 1].plot(
            z_plot[0::9] * 1e3,
            comsol_phi_s_cp[0::9],
            "o",
            color=color,
            fillstyle="none",
        )
        ax[0, 1].plot(
            z_plot * 1e3, pybamm_phi_s_cp, "-", color=color, label="{:.0f} s".format(t)
        )
        error = np.abs(pybamm_phi_s_cp - comsol_phi_s_cp)
        ax[1, 1].plot(z_plot * 1e3, error, "-", color=color)

    # force scientific notation outside 10^{+-2}
    ax[0, 0].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")
    ax[0, 1].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")
    ax[1, 0].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")
    ax[1, 1].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")

    # set ticks
    ax[0, 0].tick_params(which="both")
    ax[0, 1].tick_params(which="both")
    ax[1, 0].tick_params(which="both")
    ax[1, 1].tick_params(which="both")

    # set labels
    if sharex is False:
        ax[0, 0].set_xlabel(r"$z^*$ [mm]")
    ax[0, 0].set_ylabel(r"$\phi^*_{\mathrm{s,cn}}$ [V]")
    if sharex is False:
        ax[0, 1].set_xlabel(r"$z^*$ [mm]")
    ax[0, 1].set_ylabel(r"$\phi^*_{\mathrm{s,cp}}$ [V]")
    ax[1, 0].set_xlabel(r"$z^*$ [mm]")
    ax[1, 0].set_ylabel(r"$\phi^*_{\mathrm{s,cn}}$ (difference) [V]")
    ax[1, 1].set_xlabel(r"$z^*$ [mm]")
    ax[1, 1].set_ylabel(r"$\phi^*_{\mathrm{s,cp}}$ (difference) [V]")

    ax[0, 0].text(-0.1, 1.1, "(a)", transform=ax[0, 0].transAxes)
    ax[0, 1].text(-0.1, 1.1, "(b)", transform=ax[0, 1].transAxes)
    ax[1, 0].text(-0.1, 1.1, "(c)", transform=ax[1, 0].transAxes)
    ax[1, 1].text(-0.1, 1.1, "(d)", transform=ax[1, 1].transAxes)

    ax[0, 0].legend(
        bbox_to_anchor=(0, 1.2, 1.0, 0.102),
        loc="lower left",
        borderaxespad=0.0,
        ncol=2,
        mode="expand",
    )
    ax[0, 1].legend(
        bbox_to_anchor=(0, 1.2, 1.0, 0.102),
        loc="lower left",
        borderaxespad=0.0,
        ncol=3,
        mode="expand",
    )
    # plt.tight_layout()


def plot_cc_current_temperature(
    pybamm_model, comsol_model, mesh, solution, param, plot_times=None, sharex=False
):

    # Get discharge timescale
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)

    # Set plot times if not provided
    if plot_times is None:
        plot_times = solution.t * tau

    # Process variables
    z_plot = comsol_model.z_interp  # dimensional
    L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)

    pybamm_current_fun = pybamm.ProcessedVariable(
        pybamm_model.variables["Current collector current density [A.m-2]"],
        solution.t,
        solution.y,
        mesh=mesh,
    )
    pybamm_temp_fun = pybamm.ProcessedVariable(
        pybamm_model.variables["X-averaged cell temperature [K]"],
        solution.t,
        solution.y,
        mesh=mesh,
    )
    comsol_current_fun = pybamm.ProcessedVariable(
        comsol_model.variables["Current collector current density [A.m-2]"],
        solution.t,
        solution.y,
        mesh=mesh,
    )
    comsol_temp_fun = pybamm.ProcessedVariable(
        comsol_model.variables["X-averaged cell temperature [K]"],
        solution.t,
        solution.y,
        mesh=mesh,
    )

    # Make plot
    fig, ax = plt.subplots(2, 2, sharex=sharex, figsize=(6.4, 4))
    fig.subplots_adjust(
        left=0.1, bottom=0.1, right=0.95, top=0.85, wspace=0.3, hspace=0.5
    )
    cmap = plt.get_cmap("inferno")

    # Loop over plot_times
    for ind, t in enumerate(plot_times):
        color = cmap(float(ind) / len(plot_times))

        pybamm_current = pybamm_current_fun(z=z_plot / L_z, t=t / tau)
        comsol_current = comsol_current_fun(z=z_plot / L_z, t=t / tau)

        # plot temp wrt T0
        pybamm_temp = pybamm_temp_fun(z=z_plot / L_z, t=t / tau) - param.evaluate(
            pybamm.standard_parameters_lithium_ion.T_init_dim
        )
        comsol_temp = comsol_temp_fun(z=z_plot / L_z, t=t / tau) - param.evaluate(
            pybamm.standard_parameters_lithium_ion.T_init_dim
        )

        ax[0, 0].plot(
            z_plot[0::9] * 1e3,
            comsol_current[0::9],
            "o",
            color=color,
            fillstyle="none",
            label="COMSOL" if ind == 0 else "",
        )
        ax[0, 0].plot(
            z_plot * 1e3,
            pybamm_current,
            "-",
            color=color,
            label="PyBaMM" if ind == 0 else "",
        )
        error = np.abs(pybamm_current - comsol_current)
        ax[1, 0].plot(z_plot * 1e3, error, "-", color=color)
        ax[0, 1].plot(
            z_plot[0::9] * 1e3, comsol_temp[0::9], "o", color=color, fillstyle="none"
        )
        ax[0, 1].plot(
            z_plot * 1e3, pybamm_temp, "-", color=color, label="{:.0f} s".format(t)
        )
        error = np.abs(pybamm_temp - comsol_temp)
        ax[1, 1].plot(z_plot * 1e3, error, "-", color=color)

    # force scientific notation outside 10^{+-2}
    ax[0, 0].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")
    ax[0, 1].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")
    ax[1, 0].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")
    ax[1, 1].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")

    # set ticks
    ax[0, 0].tick_params(which="both")
    ax[0, 1].tick_params(which="both")
    ax[1, 0].tick_params(which="both")
    ax[1, 1].tick_params(which="both")

    # set labels
    if sharex is False:
        ax[0, 0].set_xlabel(r"$z^*$ [mm]")
    ax[0, 0].set_ylabel(r"$\mathcal{I}^*$ [A/m${}^2$]")
    if sharex is False:
        ax[0, 1].set_xlabel(r"$z^*$ [mm]")
    ax[0, 1].set_ylabel(r"$\bar{T}^* - \bar{T}^*_0$ [K]")
    ax[1, 0].set_xlabel(r"$z^*$ [mm]")
    ax[1, 0].set_ylabel(r"$\mathcal{I}^*$ (difference) [A/m${}^2$]")
    ax[1, 1].set_xlabel(r"$z^*$ [mm]")
    ax[1, 1].set_ylabel(r"$\bar{T}^*$ (difference) [K]")

    ax[0, 0].text(-0.1, 1.1, "(a)", transform=ax[0, 0].transAxes)
    ax[0, 1].text(-0.1, 1.1, "(b)", transform=ax[0, 1].transAxes)
    ax[1, 0].text(-0.1, 1.1, "(c)", transform=ax[1, 0].transAxes)
    ax[1, 1].text(-0.1, 1.1, "(d)", transform=ax[1, 1].transAxes)

    ax[0, 0].legend(
        bbox_to_anchor=(0, 1.2, 1.0, 0.102),
        loc="lower left",
        borderaxespad=0.0,
        ncol=2,
        mode="expand",
    )
    ax[0, 1].legend(
        bbox_to_anchor=(0, 1.2, 1.0, 0.102),
        loc="lower left",
        borderaxespad=0.0,
        ncol=3,
        mode="expand",
    )
    # plt.tight_layout()


def plot_tz_var(
    t_plot,
    z_plot,
    t_slices,
    var_name,
    units,
    comsol_var_fun,
    pybamm_var_fun,
    pybamm_bar_var_fun,
    param,
    cmap="viridis",
):
    # non-dim t and z
    L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
    z_plot_non_dim = z_plot / L_z
    t_non_dim = t_plot / tau
    t_slices_non_dim = t_slices / tau

    fig, ax = plt.subplots(2, 2, figsize=(6.4, 4))
    fig.subplots_adjust(
        left=0.15, bottom=0.1, right=0.95, top=0.95, wspace=0.4, hspace=0.8
    )
    # plot comsol var
    comsol_var = comsol_var_fun(t=t_non_dim, z=z_plot_non_dim)
    comsol_var_plot = ax[0, 0].pcolormesh(
        z_plot * 1e3, t_plot, np.transpose(comsol_var), shading="gouraud", cmap=cmap
    )
    if "cn" in var_name:
        format = "%.0e"
    else:
        format = None
    fig.colorbar(
        comsol_var_plot,
        ax=ax,
        format=format,
        location="top",
        shrink=0.42,
        aspect=20,
        anchor=(0.0, 0.0),
    )

    # plot slices
    ccmap = plt.get_cmap("inferno")
    for ind, t in enumerate(t_slices_non_dim):
        color = ccmap(float(ind) / len(t_slices))
        comsol_var_slice = comsol_var_fun(t=t, z=z_plot_non_dim)
        pybamm_var_slice = pybamm_var_fun(t=t, z=z_plot_non_dim)
        pybamm_bar_var_slice = pybamm_bar_var_fun(t=np.array([t]), z=z_plot_non_dim)
        ax[0, 1].plot(
            z_plot * 1e3, comsol_var_slice, "o", fillstyle="none", color=color
        )
        ax[0, 1].plot(
            z_plot * 1e3,
            pybamm_var_slice,
            "-",
            color=color,
            label="{:.0f} s".format(t_slices[ind]),
        )
        ax[0, 1].plot(z_plot * 1e3, pybamm_bar_var_slice, ":", color=color)
    # add dummy points for legend of styles
    comsol_p, = ax[0, 1].plot(np.nan, np.nan, "ko", fillstyle="none")
    pybamm_p, = ax[0, 1].plot(np.nan, np.nan, "k-", fillstyle="none")
    pybamm_bar_p, = ax[0, 1].plot(np.nan, np.nan, "k:", fillstyle="none")

    # compute errors
    pybamm_var = pybamm_var_fun(t=t_non_dim, z=z_plot_non_dim)
    pybamm_bar_var = pybamm_bar_var_fun(t=t_non_dim, z=z_plot_non_dim)

    error = np.abs(comsol_var - pybamm_var)
    error_bar = np.abs(comsol_var - pybamm_bar_var)

    # plot time averaged error
    ax[1, 0].plot(z_plot * 1e3, np.mean(error, axis=1), "k-", label=r"$1+1$D")
    ax[1, 0].plot(z_plot * 1e3, np.mean(error_bar, axis=1), "k:", label=r"$1+\bar{1}$D")

    # plot z averaged error
    ax[1, 1].plot(t_plot, np.mean(error, axis=0), "k-", label=r"$1+1$D")
    ax[1, 1].plot(t_plot, np.mean(error_bar, axis=0), "k:", label=r"$1+\bar{1}$D")

    # set ticks
    ax[0, 0].tick_params(which="both")
    ax[0, 1].tick_params(which="both")
    ax[1, 0].tick_params(which="both")
    if var_name in ["$\mathcal{I}^*$"]:
        ax[1, 0].set_yscale("log")
        ax[1, 0].set_yticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-2, 1e-1, 1]
    else:
        ax[1, 0].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")
    ax[1, 1].tick_params(which="both")
    if var_name in ["$\phi^*_{\mathrm{s,cn}}$"]:
        ax[1, 0].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")
    else:
        ax[1, 1].set_yscale("log")
        ax[1, 1].set_yticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-2, 1e-1, 1]

    # set labels
    ax[0, 0].set_xlabel(r"$z^*$ [mm]")
    ax[0, 0].set_ylabel(r"$t^*$ [s]")
    ax[0, 0].set_title(r"{} {}".format(var_name, units), y=1.5)
    ax[0, 1].set_xlabel(r"$z^*$ [mm]")
    ax[0, 1].set_ylabel(r"{}".format(var_name))
    ax[1, 0].set_xlabel(r"$z^*$ [mm]")
    ax[1, 0].set_ylabel("Time-averaged" + "\n" + r"absolute error {}".format(units))
    ax[1, 1].set_xlabel(r"$t^*$ [s]")
    ax[1, 1].set_ylabel("Space-averaged" + "\n" + r"absolute error {}".format(units))

    ax[0, 0].text(-0.1, 1.6, "(a)", transform=ax[0, 0].transAxes)
    ax[0, 1].text(-0.1, 1.6, "(b)", transform=ax[0, 1].transAxes)
    ax[1, 0].text(-0.1, 1.2, "(c)", transform=ax[1, 0].transAxes)
    ax[1, 1].text(-0.1, 1.2, "(d)", transform=ax[1, 1].transAxes)

    leg1 = ax[0, 1].legend(
        bbox_to_anchor=(0, 1.1, 1.0, 0.102),
        loc="lower left",
        borderaxespad=0.0,
        ncol=3,
        mode="expand",
    )

    leg2 = ax[0, 1].legend(
        [comsol_p, pybamm_p, pybamm_bar_p],
        ["COMSOL", r"$1+1$D", r"$1+\bar{1}$D"],
        bbox_to_anchor=(0, 1.5, 1.0, 0.102),
        loc="lower left",
        borderaxespad=0.0,
        ncol=3,
        mode="expand",
    )
    ax[0, 1].add_artist(leg1)

    ax[1, 0].legend(
        bbox_to_anchor=(0.0, 1.1, 1.0, 0.102),
        loc="lower right",
        borderaxespad=0.0,
        ncol=3,
        # mode="expand",
    )
    ax[1, 1].legend(
        bbox_to_anchor=(0.0, 1.1, 1.0, 0.102),
        loc="lower right",
        borderaxespad=0.0,
        ncol=3,
        # mode="expand",
    )
    # ax[1, 0].legend(loc="best")
    # ax[1, 1].legend(loc="best")
