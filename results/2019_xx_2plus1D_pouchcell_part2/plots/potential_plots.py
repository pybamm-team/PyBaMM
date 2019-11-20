import matplotlib.pyplot as plt
import numpy as np


def plot_yz_potential(t, spmecc=None, reduced=None, full=None):

    num_of_models = 0
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

    if spmecc:
        num_of_models += 1
    if reduced:
        num_of_models += 1
    if full:
        num_of_models += 1

    fig, ax = plt.subplots(len(var_names), num_of_models)
    fig.set_figheight(9)

    if num_of_models == 1:
        fig.set_figwidth(6)
    elif num_of_models == 2:
        fig.set_figwidth(11)
    elif num_of_models == 3:
        fig.set_figwidth(15)

    fig.tight_layout()

    model_idx = 0

    if num_of_models == 1:
        ax = ax[:, np.newaxis]

    if spmecc:
        y = np.linspace(0, 1.5, 100)
        z = np.linspace(0, 1, 100)
        y_dim = y * spmecc["L_z"]
        z_dim = z * spmecc["L_z"]

        for i, var_name in enumerate(var_names):
            # just doing this because effective resistance model is
            # inconsistent with other models
            if var_name == "Local voltage [V]":
                name = "Local current collector potential difference [V]"
            else:
                name = var_name
            make_2D_plot(
                fig,
                ax[i, model_idx],
                y_dim,
                z_dim,
                np.transpose(spmecc[name](t=t, y=y, z=z)),
                plot_names[i],
                color_lim[i],
            )

        model_idx += 1

    if reduced:
        y = np.linspace(0, 1.5, 100)
        z = np.linspace(0, 1, 100)
        y_dim = y * spmecc["L_z"]
        z_dim = z * spmecc["L_z"]

        for i, var_name in enumerate(var_names):
            make_2D_plot(
                fig,
                ax[i, model_idx],
                y_dim,
                z_dim,
                np.transpose(reduced[var_name](t=t, y=y, z=z)),
                plot_names[i],
                color_lim[i],
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

