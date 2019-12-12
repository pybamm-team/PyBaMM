import matplotlib.pyplot as plt
import numpy as np


def plot_current_errors(t, spmecc=None, reduced=None, full=None):

    var_names = [
        "Local current density [A.m-2]",
    ]

    plot_names = [
        r"$\mathcal{I}$ [A.m-2]",
    ]

    # tuple (vmin, vmax) for each variable
    color_lim = [(23.96, 24.06)]

    if spmecc and reduced:
        raise NotImplementedError
    if reduced and full:
        compare_reduced_and_full(t, reduced, full, var_names, plot_names, color_lim)


def compare_reduced_and_full(t, reduced, full, var_names, plot_names, color_lim):

    fig, ax = plt.subplots(1)

    y = np.linspace(0, 1.5, 100)
    z = np.linspace(0, 1, 100)
    y_dim = y * reduced["L_z"]
    z_dim = z * reduced["L_z"]

    for i, var_name in enumerate(var_names):

        error = np.abs(reduced[var_name](t=t, y=y, z=z) - full[var_name](t=t, y=y, z=z))
        error = np.transpose(error)
        make_2D_plot(
            fig, ax, y_dim, z_dim, error, plot_names[i], color_lim[i],
        )

    plt.subplots_adjust(
        hspace=0.5, top=0.95, bottom=0.05, wspace=0.3, left=0.1, right=0.95
    )

    # relative error
    fig, ax = plt.subplots(1)

    y = np.linspace(0, 1.5, 100)
    z = np.linspace(0, 1, 100)
    y_dim = y * reduced["L_z"]
    z_dim = z * reduced["L_z"]

    for i, var_name in enumerate(var_names):

        red_sol = reduced[var_name](t=t, y=y, z=z)
        full_sol = full[var_name](t=t, y=y, z=z)

        abs_error = np.abs(red_sol - full_sol)

        variation = full_sol.max() - full_sol.min()

        rel_error = abs_error / variation
        error = np.transpose(rel_error)
        make_2D_plot(
            fig, ax, y_dim, z_dim, error, plot_names[i], color_lim[i],
        )

    plt.subplots_adjust(
        hspace=0.5, top=0.95, bottom=0.05, wspace=0.3, left=0.1, right=0.95
    )


def make_2D_plot(fig, ax, y, z, var, name, color_lim):
    # im = ax.pcolormesh(
    #     y, z, var, vmin=color_lim[0], vmax=color_lim[1], shading="gouraud"
    # )
    im = ax.pcolormesh(y, z, var, shading="gouraud")
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$z$")
    ax.set_title(name)

    fig.colorbar(im, ax=ax)

