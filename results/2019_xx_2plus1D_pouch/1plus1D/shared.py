import pybamm
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt


def make_comsol_model(comsol_variables, mesh, param, z_interp=None, thermal=True):
    "Make Comsol 'model' for comparison"

    # comsol time
    comsol_t = comsol_variables["time"]
    interp_kind = "cubic"

    # discharge timescale
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)

    # interpolate using *dimensional* space
    L_x = param.evaluate(pybamm.standard_parameters_lithium_ion.L_x)
    L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)

    if z_interp is None:
        pybamm_z = mesh["current collector"][0].nodes
        z_interp = pybamm_z * L_z

    def get_interp_fun_curr_coll(variable_name):
        """
        Interpolate in space to plotting nodes, and then create function to interpolate
        in time that can be called for plotting at any t.
        """
        comsol_x = comsol_variables[variable_name + "_x"]
        comsol_z = comsol_variables[variable_name + "_z"]
        variable = comsol_variables[variable_name]

        grid_x, grid_z = np.meshgrid(L_x, z_interp)

        # Note order of rows and cols!
        interp_var = np.zeros((len(z_interp), len(comsol_x), variable.shape[1]))
        for i in range(0, variable.shape[1]):
            interp_var[:, :, i] = interp.griddata(
                np.column_stack((comsol_x, comsol_z)),
                variable[:, i],
                (grid_x, grid_z),
                method="nearest",
            )

        # average in x
        interp_var = np.nanmean(interp_var, axis=1)

        def myinterp(t):
            return interp.interp1d(comsol_t, interp_var, axis=1, kind=interp_kind)(t)[
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

    comsol_model.z_interp = z_interp

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
        pybamm_model.variables[var], solution.t, solution.y, mesh=mesh,
    )(plot_times / tau)
    comsol_var = pybamm.ProcessedVariable(
        comsol_model.variables[var], solution.t, solution.y, mesh=mesh,
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
        pybamm_model.variables[var], solution.t, solution.y, mesh=mesh,
    )
    comsol_var_fun = pybamm.ProcessedVariable(
        comsol_model.variables[var], solution.t, solution.y, mesh=mesh,
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
    pybamm_model, comsol_model, mesh, solution, param, plot_times=None, sharex=False,
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
    fig, ax = plt.subplots(2, 2, sharex=sharex, figsize=(12, 7.5))
    cmap = plt.get_cmap("inferno")

    # Loop over plot_times
    for ind, t in enumerate(plot_times):
        color = cmap(float(ind) / len(plot_times))

        pybamm_phi_s_cn = pybamm_phi_s_cn_fun(z=z_plot / L_z, t=t / tau)
        pybamm_phi_s_cp = pybamm_phi_s_cp_fun(z=z_plot / L_z, t=t / tau)
        comsol_phi_s_cn = comsol_phi_s_cn_fun(z=z_plot / L_z, t=t / tau)
        comsol_phi_s_cp = comsol_phi_s_cp_fun(z=z_plot / L_z, t=t / tau)

        ax[0, 0].plot(
            z_plot,
            comsol_phi_s_cn,
            "o",
            color=color,
            fillstyle="none",
            label="COMSOL" if ind == 0 else "",
        )
        ax[0, 0].plot(
            z_plot,
            pybamm_phi_s_cn,
            "-",
            color=color,
            label="PyBaMM (t={:.0f} s)".format(t),
        )
        error = np.abs(pybamm_phi_s_cn - comsol_phi_s_cn)
        ax[1, 0].plot(z_plot, error, "-", color=color)
        ax[0, 1].plot(
            z_plot, comsol_phi_s_cp, "o", color=color, fillstyle="none",
        )
        ax[0, 1].plot(
            z_plot, pybamm_phi_s_cp, "-", color=color,
        )
        error = np.abs(pybamm_phi_s_cp - comsol_phi_s_cp)
        ax[1, 1].plot(z_plot, error, "-", color=color)

    # set labels
    if sharex is False:
        ax[0, 0].set_xlabel(r"$z$")
    ax[0, 0].set_ylabel(r"$\phi^*_{\mathrm{s,cn}}$ [V]")
    if sharex is False:
        ax[0, 1].set_xlabel(r"$z$")
    ax[0, 1].set_ylabel(r"$\phi^*_{\mathrm{s,cp}}$ [V]")
    ax[1, 0].set_xlabel(r"$z$")
    ax[1, 0].set_ylabel(r"$\phi^*_{\mathrm{s,cn}}$ (difference) [V]")
    ax[1, 1].set_xlabel(r"$z$")
    ax[1, 1].set_ylabel(r"$\phi^*_{\mathrm{s,cp}}$ (difference) [V]")

    ax[0, 0].text(-0.1, 1.05, "(a)", transform=ax[0, 0].transAxes)
    ax[0, 1].text(-0.1, 1.05, "(b)", transform=ax[0, 1].transAxes)
    ax[1, 0].text(-0.1, 1.05, "(c)", transform=ax[1, 0].transAxes)
    ax[1, 1].text(-0.1, 1.05, "(d)", transform=ax[1, 1].transAxes)

    ax[0, 0].legend(loc="best")
    plt.tight_layout()


def plot_cc_current_temperature(
    pybamm_model, comsol_model, mesh, solution, param, plot_times=None, sharex=False,
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
    fig, ax = plt.subplots(2, 2, sharex=sharex, figsize=(12, 7.5))
    cmap = plt.get_cmap("inferno")

    # Loop over plot_times
    for ind, t in enumerate(plot_times):
        color = cmap(float(ind) / len(plot_times))

        pybamm_current = pybamm_current_fun(z=z_plot / L_z, t=t / tau)
        pybamm_temp = pybamm_temp_fun(z=z_plot / L_z, t=t / tau)
        comsol_current = comsol_current_fun(z=z_plot / L_z, t=t / tau)
        comsol_temp = comsol_temp_fun(z=z_plot / L_z, t=t / tau)

        ax[0, 0].plot(
            z_plot,
            comsol_current,
            "o",
            color=color,
            fillstyle="none",
            label="COMSOL" if ind == 0 else "",
        )
        ax[0, 0].plot(
            z_plot,
            pybamm_current,
            "-",
            color=color,
            label="PyBaMM (t={:.0f} s)".format(t),
        )
        error = np.abs(pybamm_current - comsol_current)
        ax[1, 0].plot(z_plot, error, "-", color=color)
        ax[0, 1].plot(
            z_plot, comsol_temp, "o", color=color, fillstyle="none",
        )
        ax[0, 1].plot(
            z_plot, pybamm_temp, "-", color=color,
        )
        error = np.abs(pybamm_temp - comsol_temp)
        ax[1, 1].plot(z_plot, error, "-", color=color)

    # set labels
    if sharex is False:
        ax[0, 0].set_xlabel(r"$z$")
    ax[0, 0].set_ylabel(r"$\mathcal{I}^*$ [A/m${}^2$]")
    if sharex is False:
        ax[0, 1].set_xlabel(r"$z$")
    ax[0, 1].set_ylabel(r"$\bar{T}^*$ [K]")
    ax[1, 0].set_xlabel(r"$z$")
    ax[1, 0].set_ylabel(r"$\mathcal{I}^*$ (difference) [A/m${}^2$]")
    ax[1, 1].set_xlabel(r"$z$")
    ax[1, 1].set_ylabel(r"$\bar{T}^*$ (difference) [K]")

    ax[0, 0].text(-0.1, 1.05, "(a)", transform=ax[0, 0].transAxes)
    ax[0, 1].text(-0.1, 1.05, "(b)", transform=ax[0, 1].transAxes)
    ax[1, 0].text(-0.1, 1.05, "(c)", transform=ax[1, 0].transAxes)
    ax[1, 1].text(-0.1, 1.05, "(d)", transform=ax[1, 1].transAxes)

    ax[0, 0].legend(loc="best")
    plt.tight_layout()


def plot_tz_var(
    var,
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="viridis",
    error="both",
    scale=None,
):
    fig, ax = plt.subplots(figsize=(15, 8))

    # get z vals from comsol interp points (will be dimensional)
    z_plot = comsol_model.z_interp

    # plot pybamm solution
    L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
    z_plot_non_dim = z_plot / L_z
    t_non_dim = t_plot / tau

    pybamm_var = output_variables[var](z=z_plot_non_dim, t=t_non_dim)

    if error in ["abs", "rel"]:
        plt.subplot(131)
    elif error == "both":
        plt.subplot(221)
    pybamm_plot = plt.pcolormesh(t_plot, z_plot, pybamm_var, shading="gouraud")
    plt.axis([0, t_plot[-1], 0, z_plot[-1]])
    plt.xlabel(r"$t$")
    plt.ylabel(r"$z$")
    plt.title(r"PyBaMM: " + var)
    plt.set_cmap(cmap)
    plt.colorbar(pybamm_plot)

    # plot comsol solution
    comsol_var = comsol_model.variables[var](t=t_plot)

    if error in ["abs", "rel"]:
        plt.subplot(132)
    elif error == "both":
        plt.subplot(222)
    comsol_plot = plt.pcolormesh(t_plot, z_plot, comsol_var, shading="gouraud")
    plt.axis([0, t_plot[-1], 0, z_plot[-1]])
    plt.xlabel(r"$t$")
    plt.ylabel(r"$z$")
    plt.title(r"COMSOL: " + var)
    plt.set_cmap(cmap)
    plt.colorbar(comsol_plot)

    # plot "error"
    if error in ["abs", "rel"]:
        plt.subplot(133)
        if error == "abs":
            error = np.abs(pybamm_var - comsol_var)
            diff_plot = plt.pcolormesh(t_plot, z_plot, error, shading="gouraud")
        elif error == "rel":
            if scale is None:
                scale_val = comsol_var
            error = np.abs((pybamm_var - comsol_var) / scale_val)
            diff_plot = plt.pcolormesh(t_plot, z_plot, error, shading="gouraud",)
        plt.axis([0, t_plot[-1], 0, z_plot[-1]])
        plt.xlabel(r"$t$")
        plt.ylabel(r"$z$")
        plt.title(r"Error: " + var)
        plt.set_cmap(cmap)
        plt.colorbar(diff_plot)
    elif error == "both":
        plt.subplot(223)
        abs_error = np.abs(pybamm_var - comsol_var)
        abs_diff_plot = plt.pcolormesh(t_plot, z_plot, abs_error, shading="gouraud")
        plt.axis([0, t_plot[-1], 0, z_plot[-1]])
        plt.xlabel(r"$t$")
        plt.ylabel(r"$z$")
        plt.title(r"Error (abs): " + var)
        plt.set_cmap(cmap)
        plt.colorbar(abs_diff_plot)
        plt.subplot(224)
        if scale is None:
            scale_val = comsol_var
        rel_error = np.abs((pybamm_var - comsol_var) / scale_val)
        rel_diff_plot = plt.pcolormesh(t_plot, z_plot, rel_error, shading="gouraud",)
        plt.axis([0, t_plot[-1], 0, z_plot[-1]])
        plt.xlabel(r"$t$")
        plt.ylabel(r"$z$")
        plt.title(r"Error (rel): " + var)
        plt.set_cmap(cmap)
        plt.colorbar(rel_diff_plot)
