import matplotlib.pyplot as plt
import numpy as np


def plot_potential_errors(t, spmecc=None, reduced=None, full=None):

    var_names = [
        "Negative current collector potential [V]",
        "Positive current collector potential [V]",
        "Local voltage [V]",
    ]

    plot_names = [
        r"$\phi_{\mathregular{s,n}}^*$ [V]",
        r"$\phi_{\mathregular{s,p}}^*$ [V]",
        r"$\mathcal{V}^*$ [V]",
    ]

    # tuple (vmin, vmax) for each variable
    color_lim = [(0, 0.01), (3.7, 3.9), (3.7, 3.9)]

    if spmecc and reduced:
        raise NotImplementedError
    if reduced and full:
        compare_reduced_and_full(t, reduced, full, var_names, plot_names, color_lim)


def compare_reduced_and_full(t, reduced, full, var_names, plot_names, color_lim):

    fig, ax = plt.subplots(1, 3)

    y = np.linspace(0, 1.5, 100)
    z = np.linspace(0, 1, 100)
    y_dim = y * reduced["L_z"]
    z_dim = z * reduced["L_z"]

    for i, var_name in enumerate(var_names):

        error = np.abs(reduced[var_name](t=t, y=y, z=z) - full[var_name](t=t, y=t, z=z))
        error = np.transpose(error)
        make_2D_plot(
            fig, ax[i], y_dim, z_dim, error, plot_names[i], color_lim[i],
        )

    plt.subplots_adjust(
        hspace=0.5, top=0.95, bottom=0.05, wspace=0.3, left=0.1, right=0.95
    )


def make_2D_plot(fig, ax, y, z, var, name, color_lim):
    # im = ax.pcolormesh(y, z, var, vmin=color_lim[0], vmax=color_lim[1])
    im = ax.pcolormesh(y, z, var, shading="gouraud")
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$z$")
    ax.set_title(name)

    fig.colorbar(im, ax=ax)

