#
# Shared plotting
#
import matplotlib.pyplot as plt
import numpy as np
import pybamm
from collections import defaultdict


def plot_voltages(
    all_variables, t_eval, linestyles=None, linewidths=None, figsize=(6.4, 5)
):
    """
    all_variables is a dict of dicts of dicts of dicts.
    First key: Crate
        Second key: sigma
            Third key: model name
                Fourth key: variables name -> ProcessedVariable object
    """
    # Plot
    linestyles = linestyles or ["k-", "g--", "r:", "b-."]
    linewidths = linewidths or [1.4, 1.4, 1.4, 1.4]
    n = len(list(all_variables.values())[0])
    m = len(all_variables)
    fig, axes = plt.subplots(n, m, sharey=True, figsize=figsize)
    y_min = 0.98 * min(
        np.nanmin(variables["Battery voltage [V]"](t_eval))
        for allsigma_models_variables in all_variables.values()
        for models_variables in allsigma_models_variables.values()
        for variables in models_variables.values()
    )
    y_max = 1.02 * max(
        np.nanmax(variables["Battery voltage [V]"](t_eval))
        for allsigma_models_variables in all_variables.values()
        for models_variables in allsigma_models_variables.values()
        for variables in models_variables.values()
    )
    # Strict voltage cut-offs
    y_min = max(y_min, 10.5)
    y_max = min(y_max, 14.6)
    for k, (Crate, allsigma_models_variables) in enumerate(all_variables.items()):
        for l, (sigma, models_variables) in enumerate(
            allsigma_models_variables.items()
        ):
            labels = models_variables.keys()
            ax = axes[l, k]
            t_max = max(
                np.nanmax(model["Time [h]"](t_eval))
                for model in models_variables.values()
            )
            ax.set_xlim([0, t_max])
            ax.set_ylim([y_min, y_max])
            if l == len(allsigma_models_variables) - 1:
                ax.set_xlabel("Time [h]")
            else:
                ax.set_xticklabels([])
            if l == 0:
                ax.set_title(
                    "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
                        chr(97 + k), abs(Crate), abs(Crate) * 0.6
                    )
                )
                # # Hide the right and top spines
                # ax.spines["right"].set_visible(False)
                # ax.spines["top"].set_visible(False)
                #
                # # Only show ticks on the left and bottom spines
                # ax.yaxis.set_ticks_position("left")
                # ax.xaxis.set_ticks_position("bottom")
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            if k % m == 0:
                sigma_exponent = int(np.floor(np.log10(sigma)))
                sigma_dash = 0.05 * sigma / 8000
                ax.set_ylabel(
                    (
                        "$\\hat{{\\sigma}}_p = {}\\times 10^{{{}}}$ S/m"
                        + "\n$(\\sigma'_p={}/\\mathcal{{C}})$"
                    ).format(sigma / 10 ** sigma_exponent, sigma_exponent, sigma_dash),
                    rotation=0,
                    labelpad=50,
                )
                ax.yaxis.get_label().set_verticalalignment("center")

            # else:
            #     ax.set_yticklabels([])

            for j, variables in enumerate(models_variables.values()):
                ax.plot(
                    variables["Time [h]"](t_eval),
                    variables["Battery voltage [V]"](t_eval),
                    linestyles[j],
                )
    leg = fig.legend(labels, loc="lower center", ncol=len(labels))
    plt.subplots_adjust(
        bottom=0.17, top=0.95, left=0.28, right=0.97, hspace=0.08, wspace=0.05
    )
    leg.get_frame().set_edgecolor("k")
    return fig, axes


def plot_variable(
    all_variables,
    times,
    variable,
    limits_exceptions=None,
    yaxis="SOC",
    linestyles=None,
    linewidths=None,
    figsize=(6.4, 5),
):
    sigma = 10 * 8000
    limits_exceptions = limits_exceptions or {}
    linestyles = linestyles or ["k-", "g--", "r:", "b-."]
    linewidths = linewidths or [1.4, 1.4, 1.4, 1.4]
    n = len(times)
    m = len(all_variables)
    Crates = list(all_variables.keys())
    sigmas = list(all_variables[Crates[0]].keys())
    labels = list(all_variables[Crates[0]][sigmas[0]].keys())

    z = all_variables[Crates[0]][sigmas[0]][labels[0]]["z"](0, z=np.linspace(0, 1))[
        :, 0
    ]
    z_dim = all_variables[Crates[0]][sigmas[0]][labels[0]]["z [m]"](
        0, z=np.linspace(0, 1)
    )[:, 0]

    fig, axes = plt.subplots(n, m, figsize=figsize)

    # Default limits
    y_min = pybamm.ax_min(
        [
            np.nanmin(variables[variable](times, z=z))
            for Crate, models_variables in all_variables.items()
            for variables in models_variables[sigma].values()
        ]
    )
    y_max = pybamm.ax_max(
        [
            np.nanmax(variables[variable](times, z=z))
            for Crate, models_variables in all_variables.items()
            for variables in models_variables[sigma].values()
        ]
    )
    # Exceptions
    if "min" in limits_exceptions:
        y_min = limits_exceptions["min"]
    if "max" in limits_exceptions:
        y_max = limits_exceptions["max"]

    # Plot
    for i, (Crate, allsigma_models_variables) in enumerate(all_variables.items()):
        models_variables = allsigma_models_variables[sigma]
        for j, time in enumerate(times):
            if len(times) == 1:
                ax = axes[i]
            else:
                ax = axes[j, i]
            ax.set_xlim([z_dim[0], z_dim[-1]])
            ax.set_ylim([y_min, y_max])
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

            # Title
            if j == 0:
                ax.set_title(
                    "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
                        chr(97 + i), abs(Crate), abs(Crate) * 0.6
                    )
                )
            # x-axis
            if j == len(times) - 1:
                ax.set_xlabel("z [m]")
            else:
                ax.set_xticklabels([])

            # y-axis
            if i == 0:
                # If we only want to plot one time the y label is the variable
                if len(times) == 1:
                    ax.set_ylabel(variable)
                # Otherwise the y label is the time
                else:
                    for variables in models_variables.values():
                        try:
                            if yaxis == "SOC":
                                soc = variables["State of Charge"](time)
                                ax.set_ylabel(
                                    "{}\% SoC".format(int(soc)), rotation=0, labelpad=30
                                )
                            elif yaxis == "FCI":
                                fci = variables["Fractional Charge Input"](time)
                                ax.set_ylabel(
                                    "{}\% FCI".format(int(fci)), rotation=0, labelpad=30
                                )
                            ax.yaxis.get_label().set_verticalalignment("center")
                        except ValueError:
                            pass
            else:
                ax.set_yticklabels([])

            # Plot
            for j, variables in enumerate(models_variables.values()):
                ax.plot(z_dim, variables[variable](time, z=z), linestyles[j])
    leg = fig.legend(labels, loc="lower center", ncol=len(labels), frameon=True)
    leg.get_frame().set_edgecolor("k")
    plt.subplots_adjust(
        bottom=0.17, top=0.95, left=0.18, right=0.97, hspace=0.08, wspace=0.05
    )
    return fig, axes


def plot_variable_x_z(
    all_variables,
    time_Crate_sigma,
    variable,
    limits_exceptions=None,
    figsize=(6.4, 6.4),
):
    time, Crate, sigma = time_Crate_sigma
    sigma = 8000
    models = ["1+1D LOQS", "1+1D Composite", "1+1D Full"]
    time = 0.1
    limits_exceptions = limits_exceptions or {}
    n = 1  # len(times)
    m = 3  # len(all_variables)
    models_variables = all_variables[Crate][sigma]
    fig, axes = plt.figure(figsize=figsize)

    x = models_variables[models[0]]["x"](0, x=np.linspace(0.1, 0.9))[:, 0]
    x_dim = models_variables[models[0]]["x [m]"](0, x=np.linspace(0.1, 0.9))[:, 0]
    z = models_variables[models[0]]["z"](0, z=np.linspace(0.1, 0.9, 80))[:, 0]
    z_dim = models_variables[models[0]]["z [m]"](0, z=np.linspace(0.1, 0.9, 80))[:, 0]

    # Default limits
    v_min = pybamm.ax_min(
        [
            np.nanmin(variables[variable](time, x=x, z=z))
            for variables in models_variables.values()
        ]
    )
    v_max = pybamm.ax_max(
        [
            np.nanmax(variables[variable](time, x=x, z=z))
            for variables in models_variables.values()
        ]
    )

    # Plot
    for i, model in enumerate(models):
        ax = fig.add_subplot(2, 3, i + 1)

        # Title
        ax.set_title(model)
        # x-axis
        ax.set_xlabel("x [m]")
        # y-axis
        if i == 0:
            ax.set_ylabel("z [m]")
        else:
            ax.set_yticklabels([])
        CS = ax.contourf(
            x_dim,
            z_dim,
            np.transpose(models_variables[model][variable](time, x=x, z=z)),
            vmin=v_min,
            vmax=v_max,
            levels=100,
        )
        for c in CS.collections:
            c.set_edgecolor("face")
        ax1 = ax

    fig.colorbar(CS, ax=ax1)

    # Electrodes
    L_n = pybamm.lead_acid.BaseModel().default_parameter_values.process_symbol(
        pybamm.standard_parameters_lead_acid.L_n
    )
    L_s = pybamm.lead_acid.BaseModel().default_parameter_values.process_symbol(
        pybamm.standard_parameters_lead_acid.L_s
    )
    # ax.plot([L_n, L_n], [0, np.max(z_dim)], "k--")

    # Plot
    plt.subplots_adjust(hspace=0.35, wspace=0.05)
    return fig


def plot_voltage_components(all_variables, t_eval, model, sigmas):
    n = int(len(sigmas) // np.sqrt(len(sigmas)))
    m = int(np.ceil(len(sigmas) / n))
    fig, axes = plt.subplots(n, m, sharey=True, figsize=(6.4, 2.3))
    labels = ["V", "$V_U$", "$V_k$", "$V_c$", "$V_o$", "$V_{cc}$"]
    overpotentials = [
        "Average battery reaction overpotential [V]",
        "Average battery concentration overpotential [V]",
        "Average battery electrolyte ohmic losses [V]",
        "Battery current collector overpotential [V]",
    ]
    y_min = 0.95 * min(
        np.nanmin(models_variables[model]["Battery voltage [V]"](t_eval))
        for models_variables in all_variables.values()
    )
    y_max = 1.05 * max(
        np.nanmax(models_variables[model]["Battery voltage [V]"](t_eval))
        for models_variables in all_variables.values()
    )
    for k, sigma in enumerate(sigmas):
        variables = all_variables[sigma][model]
        ax = axes.flat[k]

        # Set up
        t_max = np.nanmax(variables["Time [h]"](t_eval))
        ax.set_xlim([0, t_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Time [h]")
        sigma_exponent = int(np.floor(np.log10(sigma)))
        ax.set_title(
            "\\textbf{{({})}} $\\hat{{\\sigma}}_p = {}\\times 10^{{{}}}$ S/m".format(
                chr(97 + k), sigma / 10 ** sigma_exponent, sigma_exponent
            )
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        if k % m == 0:
            ax.set_ylabel("Voltage [V]")

        # Plot
        # Initialise
        # for lead-acid we multiply everything by 6 to
        time = variables["Time [h]"](t_eval)
        initial_ocv = variables["Average battery open circuit voltage [V]"](0)
        ocv = variables["Average battery open circuit voltage [V]"](t_eval)
        ax.fill_between(time, ocv, initial_ocv)
        top = ocv
        # Plot
        for overpotential in overpotentials:
            bottom = top + variables[overpotential](t_eval)
            ax.fill_between(time, bottom, top)
            top = bottom
        ax.plot(time, variables["Battery voltage [V]"](t_eval), "k--")
    leg = axes.flat[-1].legend(
        labels, bbox_to_anchor=(1.05, 0.5), loc="center left", frameon=True
    )
    leg.get_frame().set_edgecolor("k")
    fig.tight_layout()


def plot_times(models_times_and_voltages, Crate=1, linestyles=None):
    linestyles = linestyles or ["k-", "g--", "r:", "b-."]
    all_npts = defaultdict(list)
    solver_times = defaultdict(list)
    fig, ax = plt.subplots(1, 1)
    for i, (model, times_and_voltages) in enumerate(models_times_and_voltages.items()):
        for npts in times_and_voltages.keys():
            try:
                solver_time = times_and_voltages[npts][Crate][
                    "solution object"
                ].solve_time
            except KeyError:
                continue
            all_npts[model].append(npts * 3)
            solver_times[model].append(solver_time)
        ax.loglog(all_npts[model], solver_times[model], linestyles[i], label=model)
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("Solver time [s]")
    ax.legend(loc="best")
    fig.tight_layout()
