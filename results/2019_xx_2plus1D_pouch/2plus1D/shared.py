import pybamm
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt


def make_comsol_model(
    comsol_variables, mesh, param, y_interp=None, z_interp=None, thermal=True
):
    "Make Comsol 'model' for comparison"

    comsol_t = comsol_variables["time"]
    interp_kind = "cubic"

    # interpolate using *dimensional* space. Note that both y and z are scaled with L_z
    L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)
    if y_interp is None:
        pybamm_y = mesh["current collector"][0].edges["y"]
        y_interp = pybamm_y * L_z
    if z_interp is None:
        pybamm_z = mesh["current collector"][0].edges["z"]
        z_interp = pybamm_z * L_z  # np.linspace(0, L_z, 20)
    grid_y, grid_z = np.meshgrid(y_interp, z_interp)

    def get_interp_fun(variable_name):
        """
        Interpolate in space to plotting nodes, and then create function to interpolate
        in time that can be called for plotting at any t.
        """
        comsol_y = comsol_variables[variable_name + "_y"]
        comsol_z = comsol_variables[variable_name + "_z"]
        variable = comsol_variables[variable_name]

        # Note order of rows and cols!
        interp_var = np.zeros((len(z_interp), len(y_interp), variable.shape[1]))
        for i in range(0, variable.shape[1]):
            interp_var[:, :, i] = interp.griddata(
                np.column_stack((comsol_y, comsol_z)),
                variable[:, i],
                (grid_y, grid_z),
                method="linear",
            )

        def myinterp(t):
            return interp.interp1d(comsol_t, interp_var, axis=2, kind=interp_kind)(t)

        return myinterp

    # Create interpolating functions to put in comsol_model.variables dict
    def comsol_voltage(t):
        return interp.interp1d(comsol_t, comsol_variables["voltage"], kind=interp_kind)(
            t
        )

    comsol_phi_s_cn = get_interp_fun("phi_s_cn")
    comsol_phi_s_cp = get_interp_fun("phi_s_cp")
    comsol_current = get_interp_fun("current")

    # Create comsol model with dictionary of Matrix variables
    comsol_model = pybamm.BaseModel()
    comsol_model.variables = {
        "Terminal voltage [V]": comsol_voltage,
        "Negative current collector potential [V]": comsol_phi_s_cn,
        "Positive current collector potential [V]": comsol_phi_s_cp,
        "Current collector current density [A.m-2]": comsol_current,
    }

    # Add thermal variables
    if thermal:

        def comsol_vol_av_temperature(t):
            return interp.interp1d(
                comsol_t,
                comsol_variables["volume-averaged temperature"],
                kind=interp_kind,
            )(t)

        comsol_temperature = get_interp_fun("temperature")
        comsol_model.variables.update(
            {
                "X-averaged cell temperature [K]": comsol_temperature,
                "Volume-averaged cell temperature [K]": comsol_vol_av_temperature,
            }
        )

    comsol_model.y_interp = y_interp
    comsol_model.z_interp = z_interp

    return comsol_model


def plot_t_var(
    var, plot_times, comsol_model, output_variables, param, plot_error="both"
):
    # discharge timescale
    tau = param.process_symbol(
        pybamm.standard_parameters_lithium_ion.tau_discharge
    ).evaluate()

    pybamm_var = output_variables[var](t=(plot_times / tau))
    comsol_var = comsol_model.variables[var](plot_times)

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


def plot_2D_var(
    var,
    t,
    comsol_model,
    output_variables,
    param,
    cmap="viridis",
    error="both",
    scale=None,
):
    fig, ax = plt.subplots(figsize=(15, 8))

    # get y and z vals from comsol interp points (will be dimensional)
    y_plot = comsol_model.y_interp
    z_plot = comsol_model.z_interp

    # plot pybamm solution
    L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
    y_plot_non_dim = y_plot / L_z  # Note that both y and z are scaled with L_z
    z_plot_non_dim = z_plot / L_z
    t_non_dim = t / tau

    pybamm_var = np.transpose(
        output_variables[var](y=y_plot_non_dim, z=z_plot_non_dim, t=t_non_dim)
    )
    # If var is positive current collector potential compute relative to
    # voltage
    if var == "Positive current collector potential [V]":
        pybamm_var = pybamm_var - (
            output_variables["Terminal voltage [V]"](t=t_non_dim)
        )

    if error in ["abs", "rel"]:
        plt.subplot(131)
    elif error == "both":
        plt.subplot(221)
    pybamm_plot = plt.pcolormesh(y_plot, z_plot, pybamm_var, shading="gouraud")
    plt.axis([0, y_plot[-1], 0, z_plot[-1]])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"PyBaMM: " + var)
    plt.set_cmap(cmap)
    plt.colorbar(pybamm_plot)

    # plot comsol solution
    comsol_var = comsol_model.variables[var](t=t)

    # If var is positive current collector potential compute relative to
    # voltage
    if var == "Positive current collector potential [V]":
        comsol_var = comsol_var - comsol_model.variables["Terminal voltage [V]"](t=t)

    if error in ["abs", "rel"]:
        plt.subplot(132)
    elif error == "both":
        plt.subplot(222)
    comsol_plot = plt.pcolormesh(y_plot, z_plot, comsol_var, shading="gouraud")
    plt.axis([0, y_plot[-1], 0, z_plot[-1]])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"COMSOL: " + var)
    plt.set_cmap(cmap)
    plt.colorbar(comsol_plot)

    # plot "error"
    if error in ["abs", "rel"]:
        plt.subplot(133)
        if error == "abs":
            error = np.abs(pybamm_var - comsol_var)
            diff_plot = plt.pcolormesh(y_plot, z_plot, error, shading="gouraud")
        elif error == "rel":
            if scale is None:
                scale_val = comsol_var
            elif scale == "auto":
                scale_val = np.abs(np.max(comsol_var) - np.min(comsol_var))
            else:
                scale_val = scale
            error = np.abs((pybamm_var - comsol_var) / scale_val)
            vmax = 0.1  # max colorbar
            diff_plot = plt.pcolormesh(
                y_plot, z_plot, error, shading="gouraud", vmin=0, vmax=vmax
            )
        plt.axis([0, y_plot[-1], 0, z_plot[-1]])
        plt.xlabel(r"$y$")
        plt.ylabel(r"$z$")
        plt.title(r"Error: " + var)
        plt.set_cmap(cmap)
        plt.colorbar(diff_plot)
    elif error == "both":
        plt.subplot(223)
        abs_error = np.abs(pybamm_var - comsol_var)
        abs_diff_plot = plt.pcolormesh(y_plot, z_plot, abs_error, shading="gouraud")
        plt.axis([0, y_plot[-1], 0, z_plot[-1]])
        plt.xlabel(r"$y$")
        plt.ylabel(r"$z$")
        plt.title(r"Error (abs): " + var)
        plt.set_cmap(cmap)
        plt.colorbar(abs_diff_plot)
        plt.subplot(224)
        if scale is None:
            scale_val = comsol_var
        elif scale == "auto":
            scale_val = np.abs(np.max(comsol_var) - np.min(comsol_var))
        else:
            scale_val = scale
        rel_error = np.abs((pybamm_var - comsol_var) / scale_val)
        vmax = 0.1  # max colorbar
        rel_diff_plot = plt.pcolormesh(
            y_plot, z_plot, rel_error, shading="gouraud", vmin=0, vmax=vmax
        )
        plt.axis([0, y_plot[-1], 0, z_plot[-1]])
        plt.xlabel(r"$y$")
        plt.ylabel(r"$z$")
        plt.title(r"Error (rel): " + var)
        plt.set_cmap(cmap)
        plt.colorbar(rel_diff_plot)


def plot_cc_potentials(
    t, comsol_model, output_variables, param,
):

    # get y and z vals from comsol interp points (will be dimensional)
    y_plot = comsol_model.y_interp
    z_plot = comsol_model.z_interp

    # get pybamm potentials
    L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
    y_plot_non_dim = y_plot / L_z  # Note that both y and z are scaled with L_z
    z_plot_non_dim = z_plot / L_z
    t_non_dim = t / tau

    pybamm_phi_s_cn = np.transpose(
        output_variables["Negative current collector potential [V]"](
            y=y_plot_non_dim, z=z_plot_non_dim, t=t_non_dim
        )
    )
    pybamm_phi_s_cp = np.transpose(
        output_variables["Positive current collector potential [V]"](
            y=y_plot_non_dim, z=z_plot_non_dim, t=t_non_dim
        )
    ) - output_variables["Terminal voltage [V]"](t=t_non_dim)

    # get comsol potentials
    comsol_phi_s_cn = comsol_model.variables[
        "Negative current collector potential [V]"
    ](t=t)
    comsol_phi_s_cp = comsol_model.variables[
        "Positive current collector potential [V]"
    ](t=t) - comsol_model.variables["Terminal voltage [V]"](t=t)

    # compute difference
    diff_phi_s_cn = np.abs(pybamm_phi_s_cn - comsol_phi_s_cn)
    diff_phi_s_cp = np.abs(pybamm_phi_s_cp - comsol_phi_s_cp)

    # Make plot
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 6))
    fig.subplots_adjust(
        left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.3, hspace=0.5
    )
    cmap_n = plt.get_cmap("cividis")
    cmap_p = plt.get_cmap("viridis")

    plot_phi_s_cn = ax[0, 0].pcolormesh(
        y_plot * 1e3, z_plot * 1e3, pybamm_phi_s_cn, shading="gouraud", cmap=cmap_n
    )
    plt.colorbar(plot_phi_s_cn, ax=ax[0, 0], format="%.0e")
    plot_phi_s_cp = ax[0, 1].pcolormesh(
        y_plot * 1e3, z_plot * 1e3, pybamm_phi_s_cp, shading="gouraud", cmap=cmap_p
    )
    plt.colorbar(plot_phi_s_cp, ax=ax[0, 1], format="%.0e")
    plot_diff_s_cn = ax[1, 0].pcolormesh(
        y_plot * 1e3, z_plot * 1e3, diff_phi_s_cn, shading="gouraud", cmap=cmap_n
    )
    plt.colorbar(plot_diff_s_cn, ax=ax[1, 0], format="%.0e")
    plot_diff_s_cp = ax[1, 1].pcolormesh(
        y_plot * 1e3, z_plot * 1e3, diff_phi_s_cp, shading="gouraud", cmap=cmap_p
    )
    plt.colorbar(plot_diff_s_cp, ax=ax[1, 1], format="%.0e")

    # set ticks
    ax[0, 0].tick_params(which="both")
    ax[0, 1].tick_params(which="both")
    ax[1, 0].tick_params(which="both")
    ax[1, 1].tick_params(which="both")

    # set labels
    ax[0, 0].set_xlabel(r"$y^* [mm]$")
    ax[0, 0].set_ylabel(r"$z^* [mm]$")
    ax[0, 0].set_title(r"$\phi^*_{\mathrm{s,cn}}$ [V]")
    ax[0, 1].set_xlabel(r"$y^* [mm]$")
    ax[0, 1].set_ylabel(r"$z^* [mm]$")
    ax[0, 1].set_title(r"$\phi^*_{\mathrm{s,cp}} - V^*$ [V]")
    ax[1, 0].set_xlabel(r"$y^* [mm]$")
    ax[1, 0].set_ylabel(r"$z^* [mm]$")
    ax[1, 0].set_title(r"$\phi^*_{\mathrm{s,cn}}$ (difference) [V]")
    ax[1, 1].set_xlabel(r"$y^* [mm]$")
    ax[1, 1].set_ylabel(r"$z^* [mm]$")
    ax[1, 1].set_title(r"$\phi^*_{\mathrm{s,cp}}$ (difference) [V]")

    ax[0, 0].text(-0.1, 1.1, "(a)", transform=ax[0, 0].transAxes)
    ax[0, 1].text(-0.1, 1.1, "(b)", transform=ax[0, 1].transAxes)
    ax[1, 0].text(-0.1, 1.1, "(c)", transform=ax[1, 0].transAxes)
    ax[1, 1].text(-0.1, 1.1, "(d)", transform=ax[1, 1].transAxes)


def plot_cc_current(
    t, comsol_model, output_variables, param,
):

    # get y and z vals from comsol interp points (will be dimensional)
    y_plot = comsol_model.y_interp
    z_plot = comsol_model.z_interp

    # get pybamm current
    L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
    y_plot_non_dim = y_plot / L_z  # Note that both y and z are scaled with L_z
    z_plot_non_dim = z_plot / L_z
    t_non_dim = t / tau

    pybamm_current = np.transpose(
        output_variables["Current collector current density [A.m-2]"](
            y=y_plot_non_dim, z=z_plot_non_dim, t=t_non_dim
        )
    )

    # get comsol current
    comsol_current = comsol_model.variables[
        "Current collector current density [A.m-2]"
    ](t=t)

    # compute difference
    diff_current = np.abs(pybamm_current - comsol_current)

    # Make plot
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3))
    fig.subplots_adjust(
        left=0.1, bottom=0.2, right=0.95, top=0.8, wspace=0.3,
    )
    cmap = plt.get_cmap("plasma")

    plot_current = ax[0].pcolormesh(
        y_plot * 1e3, z_plot * 1e3, pybamm_current, shading="gouraud", cmap=cmap
    )
    plt.colorbar(plot_current, ax=ax[0])
    plot_diff_current = ax[1].pcolormesh(
        y_plot * 1e3, z_plot * 1e3, diff_current, shading="gouraud", cmap=cmap
    )
    plt.colorbar(plot_diff_current, ax=ax[1])

    # set ticks
    ax[0].tick_params(which="both")
    ax[1].tick_params(which="both")

    # set labels
    ax[0].set_xlabel(r"$y^* [mm]$")
    ax[0].set_ylabel(r"$z^* [mm]$")
    ax[0].set_title(r"$\mathcal{I}^*$ [A/m${}^2$]")
    ax[1].set_xlabel(r"$y^* [mm]$")
    ax[1].set_ylabel(r"$z^* [mm]$")
    ax[1].set_title(r"$\mathcal{I}^*$ (difference) [A/m${}^2$]")

    ax[0].text(-0.1, 1.1, "(a)", transform=ax[0].transAxes)
    ax[1].text(-0.1, 1.1, "(b)", transform=ax[1].transAxes)
